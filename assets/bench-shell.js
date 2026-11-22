// Blue Onyx Prism: the Benchmark shell of every step (templates/benchmark/shell.html), loaded after
// bench-core.js. window.BenchShell:
// - Old links: /benchmark#tab-tune, #m-<model>, ... go to their new page (BenchCore.mapHash).
// - The goal chip opens the Goal dialog (a native modal <dialog>: the rest of the page is inert,
//   Esc closes it, the focus goes back to the chip). "Try without saving" reloads the page with
//   the goal in its query (`?goal=f2&budget=500`); every view reads it (BenchShell.tryQuery()) and
//   the shell shows the "trying" banner with Save / Back. Save posts POST /v1/benchmark/goal
//   (and POST /v1/benchmark/settings for the grade's accuracy weight).
// - The status area: the run's progress from the run poll (GET /v1/benchmark?only=run, BenchPoll:
//   every 1.5 s while a run goes, every 20 s otherwise; PrismPoll runs nothing while the tab is
//   hidden), Cancel, and the "labels changed" banner with Re-score (POST /v1/benchmark/rescore).
// - Events for the page modules: onPoll(fn(step)) after each poll answer, onView(fn(why)) when the
//   saved results or the config changed (a run ended, a re-score, a save): reload the view.
(function () {
  "use strict";
  if (typeof document === "undefined") return;
  var BC = window.BenchCore, PB = window.PrismBench, el = PB.el, fmt = BC.fmt;
  var $ = function (id) { return document.getElementById(id); };
  var root = $("bench"); if (!root) return;
  var step = root.dataset.step;

  // ---- old addresses -----------------------------------------------------------------------------
  var moved = BC.mapHash(location.hash);
  if (moved) { location.replace(moved); return; }

  // ---- a goal tried without saving -----------------------------------------------------------------
  var tried = BC.tryParams(location.search);
  function tryQuery() { return tried.length ? BC.encodePairs(tried) : ""; }
  // `url` with the tried goal's fields added (an API query or a link).
  function withGoal(url) { return BC.withTry(url, tried); }
  if (tried.length) {
    document.querySelectorAll(".bn-stepper a[href^='/benchmark/']").forEach(function (a) { a.href = withGoal(a.getAttribute("href")); });
    var chip = $("goal-chip"); chip.classList.add("trying"); $("goal-chip-k").textContent = "Trying:";
    $("bn-trying").hidden = false;
    $("bn-trying-text").textContent = "You are trying another goal; nothing is saved.";
    $("bn-try-back").href = BC.withoutTry(location.pathname + location.search + location.hash);
    // The chip shows the saved goal until the view names the tried one.
    $("bn-try-back").textContent = "Back to " + $("goal-chip-v").textContent.trim();
    $("bn-try-save").addEventListener("click", function () { saveGoal(tried, null, $("bn-try-save")); });
  }
  // The goal the page shows (the views pass the server's view: summary and fields).
  var shownGoal = null;
  function setGoal(g) {
    if (!g) return;
    shownGoal = g;
    $("goal-chip-v").textContent = g.summary;
    if (tried.length) $("bn-trying-text").textContent = "You are trying " + g.summary + "; nothing is saved.";
  }

  // ---- messages ----------------------------------------------------------------------------------
  var msg = $("bn-shell-msg");
  function say(text, kind) { msg.className = "bn-msg small" + (kind === "error" ? " bn-error" : kind === "ok" ? " notice" : ""); msg.textContent = text || ""; }

  // ---- events ------------------------------------------------------------------------------------
  var pollListeners = [], viewListeners = [];
  function onPoll(fn) { pollListeners.push(fn); if (last) fn(last); }
  function onView(fn) { viewListeners.push(fn); }
  function viewChanged(why) { viewListeners.forEach(function (fn) { try { fn(why); } catch (e) { /* one view's error keeps the others */ } }); }

  // ---- the run poll --------------------------------------------------------------------------------
  var last = null, polled = null, refetch = false;
  function pollOnce() {
    var uri = BenchPoll.runUri(refetch ? null : polled && polled.modelsTag);
    refetch = false;
    return PrismPoll.fetch(uri, {cache: "no-store", headers: {"Accept": "application/json"}}).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (s) {
      var st = BenchPoll.step(polled, s);
      if (st.error) throw new Error("not a run answer");
      if (st.refetch) { refetch = true; fast.kick(); return {running: true}; }
      polled = {viewTag: st.viewTag, modelsTag: st.modelsTag, models: st.models, running: st.running};
      last = st;
      showRun(st); showLabels(st.labelsChanged);
      pollListeners.forEach(function (fn) { try { fn(st); } catch (e) { /* keep polling */ } });
      if (st.ended || st.viewChanged) viewChanged(st.ended ? "run" : "view");
      if (st.running) fast.kick();
      return st;
    });
  }
  var fast = PrismPoll.create({run: pollOnce, period: 1500, max: 30000, more: function (st) { return !!(st && st.running); }});
  var slow = PrismPoll.create({run: function () { return fast.state().busy ? null : pollOnce(); }, period: 20000, max: 60000});
  slow.start();

  // ---- progress --------------------------------------------------------------------------------------
  var bar = $("bn-progress-bar"), cancel = $("bn-cancel");
  function showRun(st) {
    var box = $("bn-progress"), run = st.run || {}, p = run.progress;
    box.hidden = !st.running;
    if (!st.running) return;
    var what = "Starting";
    if (p && p.stage === "download") what = "Downloading images";
    else if (p && p.stage === "pseudo-gt") what = "Labelling sets without labels (reference model)";
    else if (p && p.model) what = "Running " + Math.min((p.runsDone || 0) + 1, p.runsTotal || 1) + " of " + (p.runsTotal || 1) + ": " + p.model + " on " + p.device;
    $("bn-progress-what").textContent = what;
    var pct = p && BC.has(p.percent) ? Math.max(0, Math.min(100, p.percent)) : 0;
    bar.style.width = pct.toFixed(1) + "%";
    var parts = [Math.round(pct) + "%"];
    if (p && BC.has(p.etaSecs)) parts.push("about " + fmt.duration(p.etaSecs) + " left");
    if (p && p.text) parts.push(p.text);
    if (p && p.priority) parts.push(p.priority);
    $("bn-progress-text").textContent = parts.join(" · ");
  }
  cancel.addEventListener("click", function () {
    cancel.disabled = true; say("Cancelling…");
    PB.post("/v1/benchmark/cancel", "").then(function (j) {
      say(j.message || (j.success ? "Cancelled: the run stops after the current image." : "Not cancelled."), j.success ? "ok" : "error");
      fast.kick();
    }).then(function () { cancel.disabled = false; });
  });

  // ---- labels changed since the run -----------------------------------------------------------------
  var labels = $("bn-labels"), rescore = $("bn-rescore");
  function showLabels(lc) {
    var n = lc && BC.has(lc.frames) ? lc.frames : 0;
    labels.hidden = step === "run" || !n || (last && last.running);
    if (labels.hidden) return;
    var bits = [];
    if (lc.changed) bits.push(lc.changed + " changed");
    if (lc.left) bits.push(lc.left + " no longer reviewed");
    if (lc.joined) bits.push(lc.joined + " newly reviewed (a run scores those)");
    $("bn-labels-text").textContent = "Labels of " + fmt.int(n) + " frame" + (n === 1 ? "" : "s") + " of Your cameras differ from this run's: " + bits.join(", ") + ".";
  }
  rescore.addEventListener("click", function () {
    rescore.disabled = true; say("Re-scoring against the current labels…");
    PB.post("/v1/benchmark/rescore", "").then(function (j) {
      if (j.success) {
        var r = j.rescored || {};
        say("Re-scored " + fmt.int(r.frames || 0) + " frames of " + fmt.int(r.runs || 0) + " runs against the current labels." + (r.needRun ? " Frames reviewed since the run need a new run." : ""), "ok");
        labels.hidden = true; viewChanged("rescore"); fast.kick();
      } else say((j.message || "Not re-scored.") + (j._status === 409 ? "" : " Nothing changed."), "error");
    }).then(function () { rescore.disabled = false; });
  });

  // ---- the Goal dialog -------------------------------------------------------------------------------
  var dlg = $("goal-dlg"), chipBtn = $("goal-chip");
  var saved = {
    optimizeFor: dlg.dataset.optimize || "f1", tuneFor: dlg.dataset.tune || "f1",
    classes: (dlg.dataset.classesChosen || "").split(",").filter(Boolean),
    budget: dlg.dataset.budget ? parseFloat(dlg.dataset.budget) : null, weight: parseFloat(dlg.dataset.weight)
  };
  var draft = null, classChoices = (dlg.dataset.classes || "").split(",").filter(Boolean), classesFetched = false;
  function current() {
    if (shownGoal) return {optimizeFor: shownGoal.optimizeFor, tuneFor: shownGoal.tuneFor, classes: (shownGoal.classes || []).slice(), budget: BC.has(shownGoal.speedBudgetMs) ? shownGoal.speedBudgetMs : null, weight: saved.weight};
    return JSON.parse(JSON.stringify(saved));
  }
  function goalRadio(g, parent) {
    var id = "goal-o-" + g.kind, parsed = BC.parseGoal(draft.optimizeFor) || {kind: "f1"};
    var on = parsed.kind === g.kind;
    var input = el("input", {type: "radio", name: "goal-kind", id: id, value: g.kind});
    input.checked = on;
    var body = el("span", {}, [el("span", {class: "gl", text: g.label}), el("span", {class: "gd", text: g.long})]);
    var label = el("label", {class: "bn-goal-opt", for: id}, [input, body]);
    if (g.target !== undefined) {
      var tid = id + "-t";
      var t = el("input", {type: "number", id: tid, step: g.step, min: g.min, max: g.max, "aria-label": g.label + ": target"});
      t.value = on && BC.has(parsed.target) ? parsed.target : g.target;
      t.addEventListener("focus", function () { input.checked = true; pick(g.kind); });
      t.addEventListener("input", function () { input.checked = true; draft.optimizeFor = BC.goalSpec(g.kind, parseFloat(t.value)); });
      body.appendChild(el("span", {class: "gt"}, [el("span", {text: g.prefix + " ="}), t]));
    }
    input.addEventListener("change", function () { if (input.checked) pick(g.kind); });
    parent.appendChild(label);
  }
  function pick(kind) {
    var t = $("goal-o-" + kind + "-t");
    draft.optimizeFor = BC.goalSpec(kind, t ? parseFloat(t.value) : null);
    var g = BC.goalOf(kind);
    $("goal-tunefor").hidden = !g.rankOnly;
    $("goal-weight-row").hidden = kind !== "grade";
  }
  function renderClasses() {
    var box = $("goal-classes"); box.replaceChildren();
    var list = classChoices.slice();
    draft.classes.forEach(function (c) { if (list.indexOf(c) < 0) list.push(c); });
    if (!list.length) { box.appendChild(el("span", {class: "muted small", text: "Run a benchmark to list the classes its images label."})); return; }
    list.forEach(function (c) {
      var cb = el("input", {type: "checkbox", value: c}); cb.checked = draft.classes.indexOf(c) >= 0;
      cb.addEventListener("change", function () {
        draft.classes = draft.classes.filter(function (x) { return x !== c; });
        if (cb.checked) draft.classes.push(c);
      });
      box.appendChild(el("label", {class: "check"}, [cb, " " + c]));
    });
  }
  function openGoal() {
    draft = current();
    var main = $("goal-main"), more = $("goal-more"); main.replaceChildren(); more.replaceChildren();
    BC.GOALS.forEach(function (g) { goalRadio(g, g.more ? more : main); });
    var kind = (BC.parseGoal(draft.optimizeFor) || {kind: "f1"}).kind;
    $("goal-more-d").open = !!(BC.goalOf(kind) || {}).more;
    pick(kind);
    $("goal-tf").value = BC.TUNE_FOR.indexOf(draft.tuneFor) >= 0 ? draft.tuneFor : "f1";
    $("goal-weight").value = BC.has(draft.weight) ? draft.weight : 0.6;
    $("goal-budget-on").checked = BC.has(draft.budget);
    if (BC.has(draft.budget)) $("goal-budget").value = draft.budget;
    $("goal-err").hidden = true;
    if (!classesFetched && root.dataset.hasResults === "true") {
      // The classes the results are scored on (the config's list until they answer).
      classesFetched = true;
      $("goal-classes").replaceChildren(el("span", {class: "muted small", text: "Loading the classes the results are scored on…"}));
      PB.getJSON(withGoal("/v1/benchmark/compare")).then(function (j) {
        var s = j && j.compare && j.compare.set;
        if (s && s.scoredClasses && s.scoredClasses.length) classChoices = s.scoredClasses.slice();
        if (dlg.open) renderClasses();
      });
    } else renderClasses();
    dlg.showModal();
    var checked = dlg.querySelector("input[name=goal-kind]:checked"); if (checked) checked.focus();
  }
  // The dialog's goal as query pairs, or null with the problem shown.
  function readDraft() {
    var kind = (BC.parseGoal(draft.optimizeFor) || {kind: "f1"}).kind, g = BC.goalOf(kind);
    var t = $("goal-o-" + kind + "-t"), target = t ? parseFloat(t.value) : null;
    var problem = BC.targetProblem(kind, target), err = $("goal-err");
    var budgetOn = $("goal-budget-on").checked, budget = parseFloat($("goal-budget").value);
    if (!problem && budgetOn && !(budget > 0)) problem = "Speed budget: enter a time in ms above 0, or untick it.";
    var weight = parseFloat($("goal-weight").value);
    if (!problem && kind === "grade" && !(weight >= 0 && weight <= 1)) problem = "Accuracy weight: use a value from 0 to 1.";
    if (problem) { err.textContent = problem; err.hidden = false; return null; }
    err.hidden = true;
    return {pairs: BC.goalPairs({optimizeFor: BC.goalSpec(kind, target), tuneFor: g.rankOnly ? $("goal-tf").value : (draft.tuneFor || "f1"), classes: draft.classes, budget: budgetOn ? budget : null}),
      weight: kind === "grade" && Math.abs(weight - saved.weight) > 1e-9 ? weight : null};
  }
  function saveGoal(pairs, weight, button) {
    button.disabled = true; say("Saving the goal…");
    var body = new URLSearchParams(); pairs.forEach(function (p) { body.append(p[0] === "goal" ? "optimize_for" : p[0], p[1]); });
    PB.post("/v1/benchmark/goal", body).then(function (j) {
      if (!j.success) throw j;
      if (weight === null) return j;
      return PB.post("/v1/benchmark/settings", new URLSearchParams({accuracy_weight: String(weight)})).then(function (k) { if (!k.success) throw k; return j; });
    }).then(function () {
      say("Goal saved.", "ok");
      PrismPoll.navigate(BC.withoutTry(location.pathname + location.search + location.hash));
    }, function (j) {
      var text = (j && j.message) || "The goal was not saved.";
      if (dlg.open) { $("goal-err").textContent = text + " Nothing changed."; $("goal-err").hidden = false; } else say(text + " Nothing changed.", "error");
      button.disabled = false;
    });
  }
  chipBtn.addEventListener("click", openGoal);
  $("goal-x").addEventListener("click", function () { dlg.close(); });
  $("goal-cancel").addEventListener("click", function () { dlg.close(); });
  dlg.addEventListener("close", function () { chipBtn.focus(); });
  $("goal-try").addEventListener("click", function () {
    var d = readDraft(); if (!d) return;
    dlg.close();
    location.assign(BC.withTry(location.pathname + location.search, d.pairs) + location.hash);
  });
  $("goal-save").addEventListener("click", function () { var d = readDraft(); if (d) saveGoal(d.pairs, d.weight, $("goal-save")); });
  $("goal-form").addEventListener("submit", function (e) { e.preventDefault(); });

  // The image set the page shows: the step links carry it (Compare, Tune and Images open on the
  // same set), with the tried goal.
  function setSet(id) {
    document.querySelectorAll(".bn-stepper a[data-step]").forEach(function (a) {
      var st = a.dataset.step; if (st === "run" || st === "history") return;
      var u = new URL(a.getAttribute("href"), location.href);
      if (id) u.searchParams.set("set", id); else u.searchParams.delete("set");
      a.setAttribute("href", u.pathname + u.search);
    });
  }
  window.BenchShell = {
    step: step, tryQuery: tryQuery, withGoal: withGoal, trying: function () { return tried.length > 0; },
    setGoal: setGoal, setSet: setSet, say: say, onPoll: onPoll, onView: onView, viewChanged: viewChanged,
    kick: function () { fast.kick(); }, last: function () { return last; }
  };
})();
