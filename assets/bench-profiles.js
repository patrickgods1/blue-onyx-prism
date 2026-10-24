// Blue Onyx Prism: Benchmark page, steps 2 Profiles and 3 Apply (PrismBench.profiles).
// Profiles: one card per model (id "m-<model>") with one row per device it ran on: speed, finds,
// correct, the best confidence for the saved goal, the saved profile and a "Run on" radio (the
// choice is remembered per browser, PrismBench.choice). Apply: the plain list of what saving
// changes (built from status.profiles and the choices), one POST to
// /v1/benchmark/apply-profiles per button, the answer with Undo (the server's `undo` pairs) and
// "Restart now" only when the answer says needsRestart.
// Loaded after bench-glossary.js and bench-table.js, before benchmark.js, which calls
// render(status) on every poll (ES5, no build step, CSP script-src 'self').
(function () {
  var PB = window.PrismBench, L = window.PrismLive, el = PB.el, has = PB.has, F = PB.fmt;
  var $ = function (id) { return document.getElementById(id); };
  var ENDPOINT = "/v1/benchmark/apply-profiles";
  var S = {j: null, busy: false, result: null};

  // ---- the "Run on" choice per model, remembered per browser ---------------------------------
  // {model (lower case): device spec}; PB.store already survives blocked storage, the extra
  // guards cover a value of the wrong shape left by an older page.
  var choice = {};
  try { var c0 = PB.store.get("choice", {}); if (c0 && typeof c0 === "object" && !Array.isArray(c0)) choice = c0; } catch (e) { choice = {}; }
  var listeners = [];
  function lc(s) { return String(s || "").toLowerCase(); }
  PB.choice = {
    get: function (model) { var v = choice[lc(model)]; return typeof v === "string" ? v : null; },
    set: function (model, device) {
      choice[lc(model)] = device;
      try { PB.store.set("choice", choice); } catch (e) { /* the choice lasts for this page only */ }
      listeners.forEach(function (fn) { try { fn(); } catch (e) { /* a view failed; the others still update */ } });
    },
    key: function () { return JSON.stringify(choice); },
    onChange: function (fn) { listeners.push(fn); }
  };

  // ---- profile helpers ------------------------------------------------------------------------
  function near(a, b) { return has(a) && has(b) && Math.abs(a - b) < 0.005; }
  function okDevices(p) { return (p.devices || []).filter(function (d) { return d.ok && !d.fellBack; }); }
  // The device the model runs on now: the loaded one, else the row marked "now".
  function nowDevice(p) { var n = (p.devices || []).find(function (d) { return d.now; }); return p.running || (n ? n.device : null); }
  // The chosen device: the stored choice when the model ran there, else the device it runs on
  // now, else the recommended one, else the first device it ran on.
  function chosen(p) {
    var ok = okDevices(p), c = PB.choice.get(p.model);
    if (c && ok.some(function (d) { return d.device === c; })) return c;
    var d = ok.find(function (x) { return x.now; }) || ok.find(function (x) { return x.recommended; }) || ok[0];
    return d ? d.device : null;
  }
  // A device change only when the choice differs from where the model runs now.
  function deviceChange(p) {
    if (!p.configured) return null;
    var c = chosen(p), now = nowDevice(p);
    return c && c !== now && c !== p.device ? c : null;
  }
  // What "all=1" writes for a model: every device whose best confidence differs from its
  // saved profile ({device, from, to}; from null = new).
  function profileChanges(p) {
    if (!p.configured) return [];
    return okDevices(p).filter(function (d) { return has(d.best) && (!has(d.saved) || !near(d.saved, d.best)); })
      .map(function (d) { return {device: d.device, from: has(d.saved) ? d.saved : null, to: d.best}; });
  }
  function profileOf(model, device) {
    var p = ((S.j && S.j.profiles) || []).find(function (x) { return lc(x.model) === lc(model); });
    var d = p && (p.devices || []).find(function (x) { return x.device === device; });
    return d ? d.saved : null;
  }
  function configuredText(p) { return p.device === "global" ? "global setting" : p.device || "auto"; }

  var STATUS = {
    saved: ["Saved", "status-saved", "Every measured device has its best confidence saved."],
    unsaved: ["Not saved", "status-unsaved", "The best confidences of this benchmark are not saved yet: save them in step 3."],
    differs: ["Differs from latest benchmark", "status-differs", "Saved profiles differ from the best confidences of the latest benchmark for your goal."],
    none: ["No confidences", "status-none", "This benchmark found no usable confidence for this model (no labels, or it finds no objects)."]
  };
  function statusChip(status) {
    var s = STATUS[status] || STATUS.none, c = el("span", "chip status-chip " + s[1], s[0]);
    c.title = s[2]; return c;
  }

  // ---- step 2: one card per model ---------------------------------------------------------------
  function th(tr, id) { var c = el("th"); c.appendChild(PB.head(id)); tr.appendChild(c); }
  function td(tr, label, content, cls) {
    var c = el("td", cls || null); c.dataset.label = label;
    if (content === null || content === undefined) c.textContent = "-";
    else if (typeof content === "string") c.textContent = content; else c.appendChild(content);
    tr.appendChild(c); return c;
  }
  function card(p, running) {
    var box = el("section", "card profile-card"); box.id = "m-" + encodeURIComponent(p.model);
    var h = el("h3", "profile-head"); h.appendChild(el("span", "mname", p.model)); h.appendChild(statusChip(p.status)); box.appendChild(h);
    var line = el("p", "muted profile-now");
    line.appendChild(document.createTextNode("Runs on now: "));
    line.appendChild(el("code", null, p.running || "not loaded"));
    line.appendChild(document.createTextNode(" (" + configuredText(p) + ")"));
    if (has(p.fallback)) {
      line.appendChild(document.createTextNode(" · devices without a profile use "));
      line.appendChild(PB.conf(p.fallback));
      line.appendChild(document.createTextNode(p.fallbackSource === "global" ? " (global default)" : " (model default)"));
    }
    box.appendChild(line);
    if (!p.configured) box.appendChild(el("p", "muted", "Not in the config: shown for comparison, nothing is saved for it."));
    var t = el("table", "bench stack profile-table"), hd = el("thead"), tr = el("tr");
    ["device", "speed", "finds", "correct", "best_conf", "saved_conf", "run_on"].forEach(function (id) { th(tr, id); });
    hd.appendChild(tr); t.appendChild(hd);
    var body = el("tbody"), pick = chosen(p);
    (p.devices || []).forEach(function (d) {
      var r = el("tr"), ok = d.ok && !d.fellBack;
      if (ok && d.device === pick) r.classList.add("hl");
      var dc = el("span", "nowrap"); dc.appendChild(el("code", null, d.device)); if (d.label) dc.title = d.label;
      if (d.recommended) { var rt = el("span", "tag rec", "Recommended"); rt.title = "The best device for this model and your goal"; dc.appendChild(rt); }
      if (d.now) { var nt = el("span", "tag now", "Now"); nt.title = "The model runs on this device now"; dc.appendChild(nt); }
      td(r, "Device", dc);
      if (!ok) {
        r.classList.add("failed");
        var e = td(r, "Result", d.fellBack ? "fell back to another device: not measured here" : (d.error || "did not run"), "error-text"); e.colSpan = 6;
        body.appendChild(r); return;
      }
      td(r, "Speed", F.ms(d.ms), "num");
      td(r, "Finds", F.pct(d.finds), "num");
      td(r, "Correct", F.pct(d.correct), "num");
      var bc = td(r, "Best confidence", has(d.best) ? PB.conf(d.best, "best") : "-", "num");
      if (!has(d.best)) bc.title = "No usable confidence on this device for your goal";
      var sc = td(r, "Saved", has(d.saved) ? PB.conf(d.saved) : "not saved", "num" + (has(d.saved) ? "" : " muted"));
      if (d.differs && has(d.saved)) { sc.classList.add("differs"); sc.title = "Saved " + F.thr(d.saved) + "; the latest benchmark's best is " + F.thr(d.best); }
      var rc = el("td", "run-on"); rc.dataset.label = "Run on";
      var i = el("input"); i.type = "radio"; i.name = "run-on-" + lc(p.model); i.value = d.device; i.checked = d.device === pick;
      i.disabled = !p.configured || !!running;
      i.setAttribute("aria-label", "Run " + p.model + " on " + d.device);
      i.addEventListener("change", function () { if (i.checked) PB.choice.set(p.model, d.device); });
      rc.appendChild(i); r.appendChild(rc);
      body.appendChild(r);
    });
    t.appendChild(body);
    var w = el("div", "scroll"); w.dataset.keep = "w:prof:" + lc(p.model); w.appendChild(t); box.appendChild(w);
    return box;
  }
  function renderCards(j) {
    var box = $("profiles"); if (!box) return;
    var list = j.profiles;
    if (list === undefined) { box.replaceChildren(el("p", "warn", "This server does not report per-device profiles yet: update it, or use Advanced.")); return; }
    if (!list.length) { box.replaceChildren(); return; }
    if (!L) { box.replaceChildren.apply(box, list.map(function (p) { return card(p, !!j.running); })); return; }
    var running = !!j.running;
    // A notice left from an older answer is not a keyed card: start over.
    if (Array.prototype.some.call(box.children, function (c) { return c.dataset.patchKey === undefined; })) box.replaceChildren();
    L.patch(box, list.map(function (p) {
      return {key: lc(p.model), sig: JSON.stringify([p, chosen(p), running]), build: function () { return card(p, running); }};
    }));
  }

  // ---- step 3: what saving changes ----------------------------------------------------------
  function plan(j) {
    var out = [];
    (j.profiles || []).forEach(function (p) {
      out.push({p: p, profiles: profileChanges(p), use: deviceChange(p), chosen: chosen(p)});
    });
    return out;
  }
  function confText(v) { return F.thr(v); }
  function renderDiff(j) {
    var box = $("ap-diff"); if (!box) return {profiles: 0, use: 0};
    box.replaceChildren();
    var items = plan(j), np = 0, nu = 0;
    if (!items.length) { box.appendChild(el("p", "empty", j.profiles === undefined ? "This server does not report profiles yet." : "Nothing to apply yet: run a benchmark first.")); return {profiles: 0, use: 0}; }
    var ul = el("ul", "change-list apply-diff");
    items.forEach(function (x) {
      var li = el("li"), p = x.p; li.appendChild(el("strong", null, p.model + ": "));
      if (!p.configured) { li.appendChild(el("span", "muted", "not in the config, skipped")); ul.appendChild(li); return; }
      var parts = x.profiles.map(function (c) { return c.device + (c.from === null ? " new " + confText(c.to) : " " + confText(c.from) + " → " + confText(c.to)); });
      np += x.profiles.length;
      if (x.use) { nu++; parts.push("device " + configuredText(p) + " → " + x.use + " (with “use chosen devices”)"); }
      else parts.push("device stays " + configuredText(p) + (nowDevice(p) && nowDevice(p) !== p.device ? " (" + nowDevice(p) + ")" : ""));
      if (!x.profiles.length && !x.use) parts.unshift("profiles up to date");
      li.appendChild(document.createTextNode(parts.join("; ")));
      ul.appendChild(li);
    });
    box.appendChild(ul);
    return {profiles: np, use: nu};
  }
  function renderApply(j) {
    var counts = renderDiff(j), running = !!j.running, b1 = $("ap-save-use"), b2 = $("ap-save");
    if (b1) { b1.disabled = running || S.busy || counts.profiles + counts.use === 0; b1.title = running ? "Available when the benchmark finishes" : ""; }
    if (b2) { b2.disabled = running || S.busy || counts.profiles === 0; b2.title = running ? "Available when the benchmark finishes" : ""; }
    // The Apply tab shows how many changes wait.
    var tab = document.querySelector('nav.wizard a[href="#tab-apply"]');
    if (tab) {
      var n = counts.profiles + counts.use, badge = tab.querySelector(".tab-count");
      if (!badge) { badge = el("span", "tab-count"); tab.appendChild(badge); }
      badge.textContent = n ? String(n) : ""; badge.hidden = !n;
      badge.setAttribute("aria-label", n + " change" + (n === 1 ? "" : "s") + " to apply");
    }
  }

  // ---- saving and the answer --------------------------------------------------------------------
  function save(withDevices) {
    var j = S.j; if (!j) return;
    var b = new URLSearchParams(); b.append("all", "1");
    if (withDevices) plan(j).forEach(function (x) { if (x.use) b.append("use", x.use + "|" + x.p.model); });
    send(b, false);
  }
  // POST a body (pairs) to apply-profiles and show the answer in `box` (the Apply result box
  // unless given: Advanced actions use the page's status box).
  function send(body, undone, box) {
    S.busy = true; if (S.j) renderApply(S.j);
    return PB.post(ENDPOINT, body).then(function (r) { S.busy = false; showResult(r, undone, box || $("ap-result")); }, function (e) {
      S.busy = false; showResult({success: false, message: "the server did not answer (" + e + "); it may be restarting, so try again."}, undone, box || $("ap-result"));
    });
  }
  function describeDevice(c) { return (c.model ? c.model + ": " : "") + "device " + (c.from || "global") + " → " + (c.to || "global"); }
  function describeProfile(c) {
    var what = (c.model ? c.model + " on " : "") + (c.device || "") + ": ";
    if (!has(c.from) && has(c.to)) return what + "new " + F.thr(c.to);
    if (has(c.from) && !has(c.to)) return what + F.thr(c.from) + " removed";
    return what + F.thr(c.from) + " → " + F.thr(c.to);
  }
  function showResult(r, undone, box) {
    if (!box) return;
    box.hidden = false; box.className = (r.success ? "notice" : "error") + " apply-box"; box.replaceChildren();
    var ch = r.changes || {}, devs = Array.isArray(ch.devices) ? ch.devices : [], profs = Array.isArray(ch.profiles) ? ch.profiles : [];
    if (devs.length || profs.length) {
      box.appendChild(el("strong", null, undone ? "Undone:" : "Saved:"));
      var ul = el("ul", "apply-list");
      devs.forEach(function (c) { ul.appendChild(el("li", null, describeDevice(c))); });
      profs.forEach(function (c) { ul.appendChild(el("li", null, describeProfile(c))); });
      box.appendChild(ul);
    } else if (r.success) box.appendChild(el("span", null, "Nothing changed: the config already has these values. "));
    // The list already says what changed; the server's message repeats it (and the Blue Iris note
    // below), so show it only for errors or when nothing is listed.
    if (r.message && (!r.success || !(devs.length || profs.length))) box.appendChild(el("span", "muted", r.message + " "));
    else if (r.success && !r.restarting && !r.needsRestart) box.appendChild(el("span", "muted", "Applies now: the next detections use these values. "));
    if (r.success && !undone && Array.isArray(r.undo) && r.undo.length) {
      var u = el("button", "small secondary", "Undo"); u.type = "button";
      u.addEventListener("click", function () {
        u.disabled = true;
        var b = new URLSearchParams(); r.undo.forEach(function (kv) { if (Array.isArray(kv) && kv.length === 2) b.append(String(kv[0]), String(kv[1])); });
        send(b, true, box);
      });
      box.appendChild(u);
    }
    restartControls(r, box);
    if (L) L.poll();
    if (PB.refresh) PB.refresh();
    if (PB.afterApply) PB.afterApply();
  }
  // "Restart now" only when the server says a change waits for a restart; follow a restart the
  // server started itself.
  function restartControls(r, box) {
    if (r.restarting) { if (L) { L.restarting(); L.afterRestart(); } return; }
    if (r.success && r.needsRestart) {
      box.appendChild(el("span", "muted", "A device change takes effect after a restart. "));
      var b = el("button", "small danger", "Restart now"); b.type = "button";
      b.addEventListener("click", function () { if (L) L.restart(b); });
      box.appendChild(b);
    }
  }

  var b1 = $("ap-save-use"), b2 = $("ap-save");
  if (b1) b1.addEventListener("click", function () { save(true); });
  if (b2) b2.addEventListener("click", function () { save(false); });
  PB.choice.onChange(function () { if (S.j) { renderCards(S.j); renderApply(S.j); } });

  function render(j) {
    S.j = j;
    renderCards(j);
    var hw = $("other-hw");
    if (hw) L.keep(hw, String(!!j.otherHardware), function () {
      hw.replaceChildren();
      if (j.otherHardware) hw.appendChild(el("p", "warn", "These results were measured on other hardware (a different CPU or GPU). Run the benchmark again on this computer before saving them."));
    });
    renderApply(j);
  }
  PB.profiles = {
    render: render, chosen: chosen, saved: profileOf,
    // One explicit profile per [model, device, value] (value null removes it), in one POST.
    setProfiles: function (list, box) {
      var b = new URLSearchParams();
      list.forEach(function (x) { b.append("profile", x[1] + "|" + (has(x[2]) ? Number(x[2]).toFixed(2) : "none") + "|" + x[0]); });
      return send(b, false, box);
    },
    showResult: showResult, restartControls: restartControls
  };
})();
