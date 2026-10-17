// Blue Onyx Prism: Benchmark page vocabulary and shared helpers (window.PrismBench).
// One vocabulary for every view: GLOSSARY (plain name, expert term, one-line meaning, which
// direction is better) and LABELS (column headers -> glossary key), the accessible tooltip
// `term(label, key)` / `tip(label, text)`, change chips, trust badges and number formats.
// Loaded before bench-charts.js, bench-compare.js and benchmark.js (ES5, no build step, CSP
// script-src 'self': no inline handlers or style attributes; positions are set through CSSOM).
(function () {
  var PB = window.PrismBench = window.PrismBench || {};

  // ---- DOM and fetch helpers -------------------------------------------------------------------
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = text; return e; }
  function has(v) { return v !== null && v !== undefined && !(typeof v === "number" && isNaN(v)); }
  function post(url, body) {
    return fetch(url, {method: "POST", body: body, headers: {"Accept": "application/json"}}).then(function (r) {
      return r.json().then(function (j) { j._status = r.status; return j; }, function () {
        return {success: false, _status: r.status, message: "the server answered HTTP " + r.status + (r.status === 403 ? " (not allowed from here: admin actions answer this computer only)" : r.status === 404 ? " (this server does not have this feature yet; update it)" : "")};
      });
    });
  }
  function getJSON(url) {
    return fetch(url, {cache: "no-store", headers: {"Accept": "application/json"}}).then(function (r) {
      return r.json().then(function (j) { j._status = r.status; return j; }, function () { return {success: false, _status: r.status, message: "the server answered HTTP " + r.status + (r.status === 404 ? " (this server does not have this feature yet; update it)" : "")}; });
    });
  }
  // localStorage wrapped: private windows and blocked storage throw or come back empty.
  var store = {
    get: function (k, dflt) { try { var v = window.localStorage.getItem("prism.bench." + k); return v === null ? dflt : JSON.parse(v); } catch (e) { return dflt; } },
    set: function (k, v) { try { window.localStorage.setItem("prism.bench." + k, JSON.stringify(v)); } catch (e) { /* storage unavailable: the choice lasts for this page only */ } }
  };

  // ---- number formats --------------------------------------------------------------------------
  // Fractions 0..1 as "87%" (or "87.3%" with dp); ms with units; confidence with 2 dp.
  var fmt = {
    pct: function (v, dp) { return has(v) ? (v * 100).toFixed(dp === undefined ? 0 : dp) + "%" : "-"; },
    ms: function (v) { return has(v) && v < 1e300 ? (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + " ms" : "-"; },
    fps: function (v) { return has(v) ? v.toFixed(1) + " images/s" : "-"; },
    thr: function (v) { return has(v) ? v.toFixed(2) : "-"; },
    // "42% in Blue Iris": Blue Iris takes whole percents (thresholds are rounded down to 0.01).
    bi: function (v) { return has(v) ? Math.round(v * 100) + "% in Blue Iris" : ""; },
    thrBi: function (v) { return has(v) ? v.toFixed(2) + " = " + Math.round(v * 100) + "% in Blue Iris" : "-"; },
    // A difference of two fractions in percentage points: "+3.1 points".
    points: function (d, dp) { return has(d) ? (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(d * 100).toFixed(dp === undefined ? 1 : dp) + " points" : "-"; },
    signedMs: function (d) { return has(d) ? (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(d).toFixed(1) + " ms" : "-"; },
    // "16 Oct 2026, 14:02" in the browser's time zone (results store RFC 3339 UTC).
    when: function (ts) { var d = ts ? new Date(ts) : null; return d && !isNaN(d) ? d.toLocaleString([], {dateStyle: "medium", timeStyle: "short"}) : (ts || ""); },
    // "13 Oct": short tags in tables (the full date and time go in the tooltip).
    date: function (ts) { var d = ts ? new Date(ts) : null; return d && !isNaN(d) ? d.toLocaleDateString([], {day: "numeric", month: "short"}) : (ts || ""); },
    day: function (ts) { var d = ts ? new Date(ts) : null; return d && !isNaN(d) ? d.toLocaleString([], {day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"}) : (ts || ""); },
    duration: function (s) { if (!has(s)) return "-"; s = Math.max(1, Math.round(s)); if (s < 90) return s + " s"; var m = Math.round(s / 60); if (m < 90) return m + " min"; return (m / 60).toFixed(1) + " h"; }
  };

  // ---- noise constants (from status.noise; defaults match the Rust constants) -------------------
  var noise = {speed_rel: 0.05, speed_abs_ms: 0.5, se_factor: 2, plateau_points: 0.2, trust_low_objects: 100, trust_medium_objects: 300};
  function setNoise(n) { if (n && typeof n === "object") Object.keys(noise).forEach(function (k) { if (typeof n[k] === "number") noise[k] = n[k]; }); }

  // ---- the vocabulary ------------------------------------------------------------------------
  // name: plain words shown in headers; term: the expert term; meaning: one line; better:
  // "higher" | "lower" | null. Meanings may be functions (they quote the noise constants).
  var GLOSSARY = {
    precision: {name: "Correct detections", term: "precision", better: "higher", meaning: "Of everything the model detected, the share that is a real object. 90% means 9 in 10 detections are right."},
    recall: {name: "Finds objects", term: "recall", better: "higher", meaning: "Of the real objects in the images, the share the model found. 87% means it missed 13 in 100."},
    f1: {name: "Balance score", term: "F1", better: "higher", meaning: "Precision and recall combined with equal weight (their harmonic mean): high only when both are high."},
    f2: {name: "Miss-fewer score", term: "F2", better: "higher", meaning: "Like F1, but finding objects (recall) counts four times as much as precision: favors fewer missed objects."},
    ap50: {name: "Detection quality", term: "AP50 (AP@0.5)", better: "higher", meaning: "Average precision over every confidence threshold; a detection counts when its box overlaps the real object by at least 50% (IoU 0.5). The standard detector score."},
    ap50_95: {name: "Box accuracy", term: "AP50-95", better: "higher", meaning: "AP averaged over overlap requirements from 50% to 95%: also rewards tight, well-placed boxes."},
    roc_auc: {name: "Alert separation", term: "ROC AUC (frame level)", better: "higher", meaning: "How well a frame's top confidence separates frames with an object from frames without one: 100% is perfect, 50% is chance. This is the alert decision Blue Iris makes."},
    tpr: {name: "Alert rate", term: "TPR (true-positive rate, frame level)", better: "higher", meaning: "Of the frames with an object, the share that would alert at this threshold."},
    fpr: {name: "False-alert rate", term: "FPR (false-positive rate, frame level)", better: "lower", meaning: "Of the frames without an object, the share that would still alert at this threshold."},
    youden: {name: "Alert minus false alert", term: "Youden's J", better: "higher", meaning: "Alert rate minus false-alert rate; the threshold that maximizes it separates the two best."},
    p50: {name: "Time per image", term: "p50 (median)", better: "lower", meaning: "The typical time for one image: half of the images took less. Whole request: decode, resize, inference and post-processing."},
    p95: {name: "Slow-image time", term: "p95", better: "lower", meaning: "95 in 100 images took less than this; shows the occasional slow image."},
    mean: {name: "Average time", term: "mean", better: "lower", meaning: "The average time per image; a few slow images pull it above the median (p50)."},
    load_ms: {name: "Load time", term: "load / compile ms", better: "lower", meaning: "Time to read, compile and prepare the model on the device: paid once when the model loads (a cache makes later loads faster)."},
    infer: {name: "Inference vs total", term: "infer vs total", better: "lower", meaning: "Inference is the model alone on the device; total is the whole request: decode, resize, inference and post-processing."},
    fps: {name: "Images per second", term: "req/s (throughput)", better: "higher", meaning: "Requests handled per second, one at a time. The service overlaps decoding and inference, so it handles more than this."},
    iou: {name: "Box overlap", term: "IoU (intersection over union)", better: null, meaning: "How much a detected box overlaps the real object: 0.5 means the shared area is half of their combined area."},
    fp_per_image: {name: "False detections per image", term: "FP / image", better: "lower", meaning: "Detections that match no real object, per image on average."},
    frame: {name: "Per frame", term: "frame level", better: null, meaning: "Scored per image (does this frame alert?) rather than per object, the way Blue Iris decides whether to alert."},
    plateau: {name: "Near-optimal range", term: "plateau", better: null, meaning: function () { return "Every confidence in this range scores within " + noise.plateau_points + " points of the best. The value shown is the middle of the widest such range, rounded down to 0.01, so small shifts of the range edges can move it by 0.01."; }},
    objective: {name: "What the confidence optimizes", term: "objective", better: null, meaning: "The score the best confidence threshold maximizes: balance (F1), fewer misses (F2), a precision target (fewer false alerts) or a frame false-alert limit."},
    grade: {name: "Grade", term: "grade A–F", better: "higher", meaning: "A is best, F worst. The overall grade mixes the accuracy and speed grades by the accuracy weight (\"What matters most\")."},
    relative: {name: "Relative grade (*)", term: "relative to a reference model", better: null, meaning: "Scored against a reference model's detections instead of human labels: it shows agreement with that model, not true accuracy."},
    agreement: {name: "Agreement", term: "agreement with the CPU", better: null, meaning: "Whether the device detects the same objects as the CPU. A device that disagrees is never recommended."},
    reference: {name: "Reference model", term: "reference model (pseudo ground truth)", better: null, meaning: "The most accurate installed model, used as stand-in labels for images without human labels."},
    threshold: {name: "Confidence threshold", term: "confidence threshold (min_confidence)", better: null, meaning: "Detections below this confidence are dropped. 0.42 means 42% in Blue Iris; each camera's minimum confidence in Blue Iris filters again."},
    trust: {name: "Trust", term: "trust", better: null, meaning: function () { return "How far a result can be trusted: High, Medium or Low. Fewer than " + noise.trust_low_objects + " labelled objects is Low, fewer than " + noise.trust_medium_objects + " Medium; labels from a reference model instead of people, a single timed run per image, or a device that fell back to another lower it."; }},
    best_tradeoff: {name: "Best trade-off", term: "Pareto front", better: null, meaning: "No other result is both faster and more accurate. The best choice among these depends on what matters most to you."},
    same: {name: "About the same", term: "within the noise floor", better: null, meaning: function () { return "Changes this small happen between identical runs. Speed changes under " + Math.round(noise.speed_rel * 100) + "% or " + noise.speed_abs_ms + " ms, accuracy changes within " + noise.se_factor + " standard errors (normal variation for that many objects), and a confidence that stays inside the old near-optimal range (" + noise.plateau_points + " points) count as about the same."; }},
    points: {name: "points", term: "percentage points", better: null, meaning: "The difference between two percentages: 87% to 90% is +3 points (not +3%)."},
    not_comparable: {name: "Not comparable", term: "not comparable", better: null, meaning: "The two runs differ in datasets, image count, reference model, hardware or runtime version, so their numbers are not compared."},
    accuracy_score: {name: "Accuracy score", term: "accuracy score (ranking)", better: "higher", meaning: "The score the ranking uses: AP50 blended with small-object recall, or the frame ROC AUC, as chosen under Run > Settings."},
    overall: {name: "Overall", term: "overall grade", better: "higher", meaning: "Accuracy and speed grades combined by the accuracy weight (\"What matters most\")."},
    speed_grade: {name: "Speed", term: "speed grade", better: "higher", meaning: "Graded from the time per image (p50, whole request)."},
    accuracy_grade: {name: "Accuracy", term: "accuracy grade", better: "higher", meaning: "Graded from the accuracy score (AP50 with small-object recall, or ROC AUC)."},
    ran: {name: "Ran", term: "run", better: null, meaning: "When this result was measured: in the latest benchmark run, or kept from an earlier one."},
    change: {name: "Change vs last run", term: "Δ vs last run", better: null, meaning: "Compared with the pinned baseline, else the previous comparable run of this model. ▲ better, ▼ worse, ● about the same, ⚠ not comparable."}
  };
  function meaning(key) { var g = GLOSSARY[key]; if (!g) return ""; return typeof g.meaning === "function" ? g.meaning() : g.meaning; }
  function tipText(key) {
    var g = GLOSSARY[key]; if (!g) return "";
    var dir = g.better === "higher" ? " Higher is better." : g.better === "lower" ? " Lower is better." : "";
    return g.name + (g.term && g.term.toLowerCase() !== g.name.toLowerCase() ? " (" + g.term + ")" : "") + ": " + meaning(key) + dir;
  }
  // Column headers: id -> [plain label, glossary key].
  var LABELS = {
    rank: ["#", null], model: ["Model", null], device: ["Device", null], runtime: ["Runtime", null],
    overall: ["Overall", "overall"], accuracy_grade: ["Accuracy", "accuracy_grade"], speed_grade: ["Speed", "speed_grade"],
    p50: ["Time per image", "p50"], p95: ["Slow images", "p95"], infer_p50: ["Model only", "infer"], mean: ["Average", "mean"],
    load_ms: ["Load time", "load_ms"], fps: ["Images/s", "fps"],
    ap50: ["Detection quality", "ap50"], ap50_95: ["Box accuracy", "ap50_95"], roc_auc: ["Alert separation", "roc_auc"],
    precision: ["Correct", "precision"], recall: ["Finds", "recall"], f1: ["Balance (F1)", "f1"], f2: ["Miss-fewer (F2)", "f2"],
    threshold: ["Confidence", "threshold"], grade: ["Grade", "grade"], agreement: ["Agreement", "agreement"],
    trust: ["Trust", "trust"], ran: ["Ran", "ran"], change: ["Δ vs last run", "change"], tradeoff: ["Trade-off", "best_tradeoff"],
    accuracy_score: ["Accuracy score", "accuracy_score"], range: ["Near-optimal range", "plateau"], best_conf: ["Best confidence", "threshold"], now_conf: ["Confidence now", "threshold"], tpr: ["Alert rate", "tpr"], fpr: ["False alerts", "fpr"]
  };
  function label(id) { return (LABELS[id] || [id])[0]; }

  // ---- tooltip -------------------------------------------------------------------------------
  // One bubble in <body> (outside every re-rendered section, so polling never removes it). A
  // `.term` button opens it on hover, keyboard focus and tap/click; Esc closes it. The
  // listeners are delegated on the document in the capture phase, so a term inside a sortable
  // header never reaches the header's click / Enter / Space handlers.
  var bubble = null, anchor = null, anchorId = null, pinned = false, quiet = null;
  function ensureBubble() {
    if (bubble) return bubble;
    bubble = el("div", "term-tip"); bubble.id = "pb-tip"; bubble.setAttribute("role", "tooltip"); bubble.hidden = true;
    document.body.appendChild(bubble); return bubble;
  }
  function textOf(b) { return b.dataset.tip || tipText(b.dataset.term); }
  function place() {
    if (!anchor || !bubble) return;
    var r = anchor.getBoundingClientRect(), vw = document.documentElement.clientWidth || window.innerWidth;
    bubble.style.maxWidth = Math.min(340, vw - 24) + "px";
    var bw = bubble.offsetWidth, bh = bubble.offsetHeight;
    var left = Math.max(8, Math.min(r.left + window.pageXOffset, window.pageXOffset + vw - bw - 8));
    var top = r.bottom + window.pageYOffset + 6;
    if (r.bottom + bh + 12 > window.innerHeight && r.top > bh + 12) top = r.top + window.pageYOffset - bh - 6;
    bubble.style.left = left + "px"; bubble.style.top = top + "px";
  }
  function show(b, stick) {
    ensureBubble();
    if (anchor && anchor !== b) anchor.removeAttribute("aria-describedby");
    anchor = b; anchorId = b.dataset.tipId; pinned = !!stick;
    bubble.textContent = textOf(b); bubble.hidden = false;
    b.setAttribute("aria-describedby", "pb-tip"); b.setAttribute("aria-expanded", "true");
    place();
  }
  function hide() {
    if (anchor) { anchor.removeAttribute("aria-describedby"); anchor.setAttribute("aria-expanded", "false"); }
    anchor = null; anchorId = null; pinned = false; if (bubble) bubble.hidden = true;
  }
  function termOf(t) { return t && t.closest ? t.closest(".term") : null; }
  document.addEventListener("click", function (e) {
    var b = termOf(e.target);
    if (b) { e.stopPropagation(); e.preventDefault(); if (anchor === b && pinned) hide(); else show(b, true); return; }
    if (anchor && bubble && !bubble.contains(e.target)) hide();
  }, true);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && anchor) { var a = anchor; hide(); if (document.activeElement !== a && a.isConnected) { quiet = a; a.focus(); quiet = null; } return; }
    var b = termOf(e.target); if (b && (e.key === "Enter" || e.key === " ")) e.stopPropagation();
  }, true);
  document.addEventListener("mouseover", function (e) { var b = termOf(e.target); if (b && b !== anchor) show(b, false); }, true);
  document.addEventListener("mouseout", function (e) {
    var b = termOf(e.target); if (!b || b !== anchor || pinned) return;
    var to = e.relatedTarget; if (to && (b.contains(to) || (bubble && bubble.contains(to)))) return;
    if (document.activeElement !== b) hide();
  }, true);
  document.addEventListener("focusin", function (e) { var b = termOf(e.target); if (b && b !== quiet) show(b, false); }, true);
  document.addEventListener("focusout", function (e) { var b = termOf(e.target); if (b && b === anchor && !pinned) hide(); }, true);
  window.addEventListener("resize", place);
  window.addEventListener("scroll", place, true);
  // After a re-render: follow the open tooltip to its rebuilt button (same tip id), else close.
  function refreshTip() {
    if (!anchor || anchor.isConnected) { place(); return; }
    var n = anchorId ? document.querySelector('.term[data-tip-id="' + anchorId.replace(/"/g, "") + '"]') : null;
    if (n) { var p = pinned; show(n, p); } else hide();
  }
  // A stable id from the label and key (so a rebuilt button is found again).
  function tipId(label, key, scope) { return (scope || "") + "|" + (key || "") + "|" + label; }
  // `label` with a tooltip from the glossary entry `key`.
  function term(label, key, scope) {
    var b = el("button", "term keep-hover", label); b.type = "button";
    if (key) b.dataset.term = key; b.dataset.tipId = tipId(label, key, scope); b.setAttribute("aria-expanded", "false");
    if (label === "?" && GLOSSARY[key]) { b.setAttribute("aria-label", "What is " + GLOSSARY[key].name.toLowerCase() + "?"); b.classList.add("term-q"); }
    return b;
  }
  // `label` with a free tooltip text (chips, badges).
  function tip(label, text, cls, scope) {
    var b = el("button", "term keep-hover" + (cls ? " " + cls : ""), label); b.type = "button";
    b.dataset.tip = text; b.dataset.tipId = tipId(label, "t:" + (scope || "") + text.slice(0, 40), scope); b.setAttribute("aria-expanded", "false");
    return b;
  }
  // A header cell content: the plain label with its glossary tooltip.
  function head(id) { var l = LABELS[id] || [id, null]; return l[1] ? term(l[0], l[1], "h") : document.createTextNode(l[0]); }

  // ---- change chips and trust badges ---------------------------------------------------------
  var CHIP = {
    better: ["▲", "better", "chip-better"], worse: ["▼", "worse", "chip-worse"],
    same: ["●", "about the same", "chip-same"], not_comparable: ["⚠", "not comparable", "chip-nc"],
    first: ["○", "first run", "chip-first"]
  };
  // A chip: icon + word + color; `word` overrides the verdict word ("faster"); `why` opens as a
  // tooltip.
  function chip(verdict, word, why) {
    var c = CHIP[verdict] || CHIP.same, text = c[0] + " " + (word || c[1]);
    if (why) return tip(text, why, "chip " + c[2]);
    return el("span", "chip " + c[2], text);
  }
  // Words for a metric's verdict.
  var WORDS = {
    p50_ms: ["faster", "slower"], infer_p50_ms: ["faster model", "slower model"], fps: ["more images/s", "fewer images/s"],
    ap50: ["more accurate", "less accurate"], ap50_95: ["tighter boxes", "looser boxes"], roc_auc: ["separates better", "separates worse"],
    precision: ["fewer false detections", "more false detections"], recall: ["finds more", "finds less"], f1: ["better balance", "worse balance"],
    threshold: ["better confidence", "worse confidence"]
  };
  var METRIC_NAMES = {p50_ms: "Time per image (p50)", infer_p50_ms: "Model-only time (infer p50)", fps: "Images per second", ap50: "Detection quality (AP50)", ap50_95: "Box accuracy (AP50-95)", roc_auc: "Alert separation (ROC AUC)", precision: "Correct detections (precision)", recall: "Finds objects (recall)", f1: "Balance (F1)", threshold: "Best confidence"};
  function metricChip(mc) {
    if (!mc) return chip("same");
    var w = WORDS[mc.metric], word = mc.verdict === "better" && w ? w[0] : mc.verdict === "worse" && w ? w[1] : null;
    return chip(mc.verdict, word, mc.why || null);
  }
  function isSpeed(m) { return m === "p50_ms" || m === "infer_p50_ms" || m === "fps"; }
  // Format a metric value: ms, images/s, % or a confidence.
  function metricValue(metric, v) {
    if (!has(v)) return "-";
    if (metric === "p50_ms" || metric === "infer_p50_ms") return fmt.ms(v);
    if (metric === "fps") return fmt.fps(v);
    if (metric === "threshold") return fmt.thr(v);
    return fmt.pct(v, 1);
  }
  function metricDelta(metric, d) {
    if (!has(d)) return "-";
    if (metric === "p50_ms" || metric === "infer_p50_ms") return fmt.signedMs(d);
    if (metric === "fps") return (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(d).toFixed(1) + " images/s";
    if (metric === "threshold") return (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(d).toFixed(2);
    return fmt.points(d);
  }
  // The overall verdict of a model's Changes: chips for speed and accuracy (of `device`, else
  // the first device listed), or one chip for "not comparable" / "first run".
  function changeChips(ch, device, metricKey) {
    var box = el("span", "chips");
    // undefined: an older server; null: the model failed (nothing to compare).
    if (!ch) return box;
    if (!ch.baseline) { box.appendChild(chip("first", null, "No earlier run of this model to compare with.")); return box; }
    var base = "vs " + (ch.baseline.pinned ? "the pinned baseline" : "the run") + " of " + fmt.when(ch.baseline.timestamp);
    if (!ch.comparable) { box.appendChild(chip("not_comparable", null, "Not compared " + base + ": " + ((ch.reasons || []).join("; ") || "the runs differ") + ".")); return box; }
    var ms = ch.metrics || [], dev = device || (ms.find(function (m) { return m.device; }) || {}).device;
    var of = function (metric) { return ms.find(function (m) { return m.metric === metric && (m.device === dev || !m.device); }); };
    var sp = of("p50_ms"), ac = of(metricKey || "ap50") || of("roc_auc") || of("ap50");
    [[sp, "speed"], [ac, "accuracy"]].forEach(function (p) {
      if (!p[0]) return;
      var c = metricChip(p[0]); c.dataset.tip = METRIC_NAMES[p[0].metric] + " " + base + ": " + metricValue(p[0].metric, p[0].before) + " → " + metricValue(p[0].metric, p[0].after) + ". " + (p[0].why || "");
      if (p[0].verdict === "same") c.textContent = "● " + p[1] + " same";
      box.appendChild(c);
    });
    if (!box.children.length) box.appendChild(chip("same", null, (ch.summary || "about the same") + " (" + base + ")."));
    return box;
  }
  var TRUST = {high: "High", medium: "Medium", low: "Low"};
  // "High / Medium / Low trust" with the reasons in a tooltip; nothing when absent.
  function trustBadge(t, scope) {
    if (!t || !t.level) return el("span", "muted", "");
    var lv = String(t.level).toLowerCase(), reasons = (t.reasons || []).filter(Boolean);
    var why = (TRUST[lv] || lv) + " trust: " + (reasons.length ? reasons.join("; ") + "." : lv === "high" ? "enough labelled objects, human labels, several timed runs per image and no device fallback." : "see the glossary.");
    return tip((TRUST[lv] || lv) + " trust", why, "trust trust-" + lv, scope);
  }
  // "this run" / "earlier · 12 Oct 16:53" for a model result.
  function runTag(m) {
    if (!m) return el("span");
    if (m.this_run) return tip("this run", "Measured in the latest benchmark run (" + fmt.when(m.timestamp) + ").", "tag run-this", "run");
    if (m.this_run === false || m.run_id !== undefined) return tip("earlier · " + fmt.date(m.timestamp), "Kept from an earlier run (" + fmt.when(m.timestamp) + "); the latest run did not include this model.", "tag run-old", "run");
    return tip(fmt.date(m.timestamp), "Measured " + fmt.when(m.timestamp) + ".", "tag run-old", "run");
  }

  // ---- glossary list (the no-hover fallback) -------------------------------------------------
  var GLOSSARY_ORDER = ["threshold", "recall", "precision", "f1", "f2", "ap50", "ap50_95", "roc_auc", "tpr", "fpr", "youden", "frame", "iou", "fp_per_image", "p50", "p95", "mean", "infer", "load_ms", "fps", "grade", "relative", "agreement", "reference", "objective", "plateau", "trust", "best_tradeoff", "same", "not_comparable", "points"];
  function glossaryDetails(key) {
    var d = el("details", "help glossary"); d.dataset.keep = "d:gloss:" + (key || "");
    d.appendChild(el("summary", null, "What do these terms mean?"));
    var dl = el("dl", "gloss-list");
    GLOSSARY_ORDER.forEach(function (k) {
      var g = GLOSSARY[k]; if (!g) return;
      var dt = el("dt", null, g.name); if (g.term && g.term.toLowerCase() !== g.name.toLowerCase()) dt.appendChild(el("span", "muted", " (" + g.term + ")"));
      dl.appendChild(dt);
      dl.appendChild(el("dd", null, meaning(k) + (g.better === "higher" ? " Higher is better." : g.better === "lower" ? " Lower is better." : "")));
    });
    d.appendChild(dl); return d;
  }

  // Runtime of a device spec: "openvino" | "onnxruntime".
  function runtimeOf(device, given) { return given || (/^ort:/.test(device || "") ? "onnxruntime" : "openvino"); }
  var RUNTIME_NAMES = {openvino: "OpenVINO", onnxruntime: "ONNX Runtime"};

  // Model -> series color 1..8, stable: alphabetical over every model the page has seen.
  var colorOrder = [];
  function colorsFor(names) { names.slice().sort(function (a, b) { return a.toLowerCase() < b.toLowerCase() ? -1 : 1; }).forEach(function (n) { if (colorOrder.indexOf(n.toLowerCase()) < 0) colorOrder.push(n.toLowerCase()); }); }
  function colorOf(name) { var i = colorOrder.indexOf(String(name).toLowerCase()); if (i < 0) { colorOrder.push(String(name).toLowerCase()); i = colorOrder.length - 1; } return (i % 8) + 1; }

  // "F2: miss fewer objects" for an objective string ("f2", "precision:0.9", "fpr:0.05").
  function objectiveName(o) {
    o = String(o || "f1"); var k = o.split(":")[0], t = parseFloat(o.split(":")[1]);
    if (k === "f1") return "F1: balanced";
    if (k === "f2") return "F2: miss fewer objects";
    if (k === "precision") return "precision \u2265 " + Math.round(t * 100) + "%: fewer false alerts";
    if (k === "recall") return "recall \u2265 " + Math.round(t * 100) + "%";
    if (k === "fpr") return "false-alert frames \u2264 " + Math.round(t * 100) + "%";
    if (k === "youden") return "Youden's J: alert rate minus false-alert rate";
    return o;
  }
  PB.objectiveName = objectiveName;
  PB.el = el; PB.has = has; PB.post = post; PB.getJSON = getJSON; PB.store = store; PB.fmt = fmt;
  PB.noise = noise; PB.setNoise = setNoise;
  PB.GLOSSARY = GLOSSARY; PB.LABELS = LABELS; PB.label = label; PB.tipText = tipText; PB.meaning = meaning;
  PB.term = term; PB.tip = tip; PB.head = head; PB.refreshTip = refreshTip; PB.hideTip = hide;
  PB.chip = chip; PB.metricChip = metricChip; PB.changeChips = changeChips; PB.trustBadge = trustBadge; PB.runTag = runTag;
  PB.metricValue = metricValue; PB.metricDelta = metricDelta; PB.METRIC_NAMES = METRIC_NAMES; PB.isSpeed = isSpeed;
  PB.glossaryDetails = glossaryDetails; PB.runtimeOf = runtimeOf; PB.RUNTIME_NAMES = RUNTIME_NAMES;
  PB.colorsFor = colorsFor; PB.colorOf = colorOf;
})();
