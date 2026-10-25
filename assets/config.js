// Blue Onyx Prism: Config page (served from /static/; no inline scripts under the CSP).
// The Browse... buttons (model, classes, folders) are wired by /static/browse.js.
// Confirmation for large downloads and removals (without JS the button label shows the size).
// Delegated: the Resources section is re-rendered in place.
document.addEventListener("click", function (e) {
  var b = e.target.closest && e.target.closest("button[data-confirm]");
  if (b && !confirm(b.dataset.confirm)) e.preventDefault();
});
// Force CPU disables the Device select (a disabled select is not submitted: the saved device stays).
(function () {
  var cb = document.getElementById("force-cpu"), sel = document.getElementById("device"), hint = document.getElementById("device-hint");
  function sync() { sel.disabled = cb.checked; hint.hidden = !cb.checked; }
  cb.addEventListener("change", sync);
})();
// The models JSON opens when the server rejected it.
(function () {
  var err = document.querySelector("p.error");
  if (err && /models/.test(err.textContent)) document.querySelector("details.advanced").open = true;
})();
// Resources: actions (Download, Remove, Add to config, Export, Cancel) post in place
// (PrismLive data-ajax forms); the section is then re-rendered from a fresh /config without
// moving the page, and /v1/resources is polled while something downloads or exports: progress
// updates in place, a row that changes state re-renders the section (no page reload).
(function () {
  var L = window.PrismLive, box = document.getElementById("resources"), sources = document.getElementById("model-sources");
  if (!L || !window.fetch || !box) return;
  var boxes = [box, sources].filter(Boolean);
  var key = function (r) { return r.busy ? "busy" : r.state; };
  var timer = null, swapping = null;
  function rowOf(id) { return box.querySelector('[data-id="' + CSS.escape(id) + '"]'); }
  // Replace the sections' content (Resources, and Add a model by path / Local models) with the
  // fresh page's, keeping scroll offsets, open <details> and the answers shown next to the buttons.
  function swapOne(doc, b) {
    var fresh = doc.getElementById(b.id);
    if (!fresh) return;
    var msgs = [];
    b.querySelectorAll(".action-msg").forEach(function (m) { var tr = m.closest("[data-id]"); msgs.push([tr ? tr.dataset.id : null, m]); });
    var st = L.view.capture(b);
    b.replaceChildren.apply(b, Array.prototype.slice.call(document.adoptNode(fresh).childNodes));
    L.view.restore(b, st);
    msgs.forEach(function (p) {
      var row = p[0] && b.querySelector('[data-id="' + CSS.escape(p[0]) + '"]');
      if (row) (row.tagName === "TR" ? row.lastElementChild : row).appendChild(p[1]);
      else { p[1].classList.add("section-msg"); b.querySelector("h2").after(p[1]); }
    });
  }
  function swap() {
    if (swapping) return swapping;
    swapping = fetch("/config", {cache: "no-store"}).then(function (r) { return r.text(); }).then(function (html) {
      var doc = new DOMParser().parseFromString(html, "text/html"), y = window.scrollY;
      boxes.forEach(function (b) { swapOne(doc, b); });
      var focusIn = boxes.some(function (b) { return document.activeElement && b.contains(document.activeElement); });
      if (Math.abs(window.scrollY - y) > 1 && !focusIn) window.scrollTo(window.scrollX, y);
    }).catch(function () {}).then(function () { swapping = null; watch(); });
    return swapping;
  }
  function tick() {
    fetch("/v1/resources", {cache: "no-store"}).then(function (r) { return r.json(); }).then(function (j) {
      var changed = false;
      j.resources.forEach(function (r) {
        var row = rowOf(r.id);
        if (!row) return;
        var was = row.dataset.busy === "true" ? "busy" : row.dataset.state;
        if (was !== key(r)) { changed = true; return; }
        if (r.busy) {
          row.querySelector(".res-text").textContent = r.state_text;
          var bar = row.querySelector("progress");
          if (bar) { if (r.progress !== null && r.progress !== undefined) bar.value = r.progress; else bar.removeAttribute("value"); }
        }
      });
      if (changed) swap(); else watch();
    }).catch(function () { watch(); });
  }
  // Poll while a row is busy.
  function watch() {
    clearTimeout(timer); timer = null;
    if (box.querySelector('tr[data-busy="true"]')) timer = setTimeout(tick, 1500);
  }
  boxes.forEach(function (b) { b.addEventListener("prism:action", function (e) {
    // A refused "Add a model by path" keeps what was typed.
    if (e.target.closest && e.target.closest("#resource-custom-model") && !(e.detail && e.detail.ok)) return;
    swap();
  }); });
  watch();
})();
// ---- Live config: refresh fields from GET /v1/config, keep and mark unsaved edits; a save
// writes the form as shown (last write wins).
(function () {
  var L = window.PrismLive;
  if (!L || !window.fetch) return;
  var U = L.ui, $ = function (id) { return document.getElementById(id); };
  var serverForm = $("server"), card = $("models"), jsonForm = $("models-json-form");
  var pageRev = +$("cfg-title").dataset.revision, epoch = null, own = {}, resetAfter = null;
  U.markEdits(serverForm); U.markEdits(card); U.markEdits(jsonForm);
  function show(v) { return v === "" || v === null || v === undefined ? "global" : v === true ? "on" : v === false ? "off" : String(v); }
  function fmtThr(t) { return t === null || t === undefined ? "" : String(t); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

  // Server form and the models JSON form: inputs/selects/checkboxes keyed by name. `reset` is
  // the form just saved (its fields take the server values) or null.
  function refreshServer(j, reset) {
    var view = j.forms.server.view;
    Object.keys(view).forEach(function (name) {
      var ctl = name === "models_json" ? jsonForm.elements[name] : serverForm.elements[name];
      if (!ctl || ctl.type === "hidden" || !ctl.tagName) return;
      if (reset === true || reset === ctl.form) { U.set(ctl, view[name]); U.note(ctl, null); ctl.classList.remove("edited"); return; }
      var describe = name === "models_json" ? function () { return "a newer models list (see the change banner)"; } : show;
      U.refresh(ctl, view[name], describe);
    });
    var adv = $("models-json"), ta = jsonForm.elements.models_json;
    if (ta && ta.classList.contains("edited-conflict")) adv.open = true;
    var dev = $("device"), fc = $("force-cpu"); if (dev && fc) { dev.disabled = fc.checked; $("device-hint").hidden = !fc.checked; }
  }

  // ---- Threshold per device. The Threshold field is for the device the row's Device option
  // resolves to (option data-key; the hidden threshold_key.<model> field tells the server).
  // Switching the device shows that device's profile (data-thr), else fills in the benchmark's
  // best there (data-bench, marked as an edit: Save stores it as that device's profile), else
  // empty with the fallback as placeholder. Values typed per (model, device) are kept while
  // switching back and forth.
  var typed = {};
  function thrParts(tr) {
    return {sel: tr.querySelector("td.model-device select"), cell: tr.querySelector("td.model-thr"), input: tr.querySelector("input.thr-input"),
      key: tr.querySelector("input.thr-key"), hint: tr.querySelector(".thr-hint"), def: tr.querySelector("input.thr-default")};
  }
  // Same device: "openvino:gpu" == "openvino:gpu.0", "ort:cuda" == "ort:cuda:0".
  function devId(d) { return String(d || "").replace(/([.:])0$/, ""); }
  function lookup(map, key) {
    if (!map || !key) return null;
    if (Object.prototype.hasOwnProperty.call(map, key)) return map[key];
    var id = devId(key), k = Object.keys(map).find(function (x) { return devId(x) === id; });
    return k === undefined ? null : map[k];
  }
  function selectedOption(sel) { return sel && sel.selectedOptions && sel.selectedOptions[0]; }
  function fallbackOf(r) { return {value: r.cell.dataset.fallback || "", src: r.cell.dataset.fallbackSrc || ""}; }
  // The hint under the field: "for openvino:cpu · benchmark best 0.34 [use]".
  function thrHint(tr, prefilled) {
    var r = thrParts(tr), o = selectedOption(r.sel);
    if (!r.hint || !o) return;
    var key = r.key.value, bench = o.dataset.bench || "", thr = o.dataset.thr || "", fb = fallbackOf(r);
    r.hint.replaceChildren();
    if (key) { var f = el("span", "muted", "for "); f.append(el("code", null, key)); r.hint.append(f); }
    if (!key || !bench) return;
    r.hint.append(" · ", el("span", "muted", "benchmark best " + bench), " ");
    var effective = thr !== "" ? thr : fb.value;
    if (prefilled) r.hint.append(el("span", "muted", "for this device; Save to keep"));
    else if (effective !== "" && parseFloat(effective) === parseFloat(bench)) {
      var running = r.cell.dataset.runningThr, saved = o.defaultSelected;
      r.hint.append(el("span", "muted", saved && running !== "" && parseFloat(running) === parseFloat(bench) ? "in use" : "set"));
    } else {
      var b = el("button", "small secondary", "use"); b.type = "button"; b.dataset.useThreshold = bench; b.title = "Put " + bench + " in the threshold field (then save)"; r.hint.append(b);
    }
  }
  // Point the field at the selected option's device. `prefill`: a user's switch (fill in the
  // benchmark's best when the device has no profile); `keepValue`: leave the value (carried over
  // from an edit), only rebase the key, baseline and hint.
  function applyDevice(tr, prefill, keepValue) {
    var r = thrParts(tr), o = selectedOption(r.sel);
    if (!r.input || !r.key || !o) return;
    var key = o.dataset.key || "", thr = o.dataset.thr || "", bench = o.dataset.bench || "", fb = fallbackOf(r);
    var mem = typed[tr.dataset.model + "\n" + key], value = r.input.value;
    r.key.value = key;
    // Without a device the field edits the model's default (as older pages did).
    var base = key ? thr : (r.def ? r.def.defaultValue : "");
    r.input.defaultValue = base;
    var prefilled = false;
    if (keepValue) r.input.value = value;
    else if (mem !== undefined) r.input.value = mem;
    else if (prefill && key && thr === "" && bench !== "") { r.input.value = bench; prefilled = true; }
    else r.input.value = base;
    r.input.placeholder = fb.value ? fb.value + " " + fb.src : "";
    r.input.classList.toggle("edited", U.edited(r.input));
    if (r.def) r.def.disabled = !key;
    // A profile marked for removal on this device empties the field.
    var rm = key && tr.querySelector('input[name^="profile_remove."][value="' + CSS.escape(key) + '"]');
    if (rm && rm.checked && !keepValue) { r.input.value = ""; r.input.classList.toggle("edited", U.edited(r.input)); }
    thrHint(tr, prefilled);
  }
  // The profiles list under "Confidence per device" (rebuilt when the saved profiles changed;
  // boxes still listed keep their state unless `force`).
  function refreshProfiles(tr, m, force) {
    var list = tr.querySelector(".thr-profile-list"), none = tr.querySelector(".thr-profile-none"), sum = tr.querySelector(".thr-profiles summary");
    if (!list) return;
    var map = m.deviceThresholds || {}, keys = Object.keys(map).sort(), name = tr.querySelector("input.thr-input").name.slice("threshold.".length);
    var sig = keys.map(function (k) { return k + "=" + Number(map[k]).toFixed(2); }).join(",");
    if (list.dataset.sig === undefined) list.dataset.sig = Array.prototype.map.call(list.querySelectorAll("li"), function (li) { return li.dataset.device + "=" + li.dataset.thr; }).join(",");
    if (list.dataset.sig !== sig || force) {
      var checked = {};
      if (!force) list.querySelectorAll("input[type=checkbox]").forEach(function (c) { checked[c.value] = c.checked; });
      list.replaceChildren();
      keys.forEach(function (k) {
        var li = el("li"); li.dataset.device = k; li.dataset.thr = Number(map[k]).toFixed(2);
        var cb = el("input"); cb.type = "checkbox"; cb.name = "profile_remove." + name; cb.value = k; cb.checked = !!checked[k];
        var lab = el("label", "check small"); lab.append(cb, " remove");
        li.append(el("code", null, k), " " + Number(map[k]).toFixed(2) + " ", lab); list.append(li);
      });
      list.dataset.sig = sig;
    }
    if (none) none.hidden = keys.length > 0;
    if (sum) sum.textContent = "Confidence per device" + (keys.length ? " (" + keys.length + ")" : "");
  }
  // One row's threshold controls from the /v1/config model `m`.
  function refreshThreshold(tr, m, j, force) {
    var r = thrParts(tr);
    if (!r.input) return;
    if (force) Object.keys(typed).forEach(function (k) { if (k.indexOf(tr.dataset.model + "\n") === 0) delete typed[k]; });
    r.cell.dataset.fallback = m.fallback ? Number(m.fallback.value).toFixed(2) : j.globalThreshold.toFixed(2);
    r.cell.dataset.fallbackSrc = m.fallback ? m.fallback.source : "global";
    r.cell.dataset.runningThr = m.running.threshold === null || m.running.threshold === undefined ? "" : Number(m.running.threshold).toFixed(2);
    var fb = fallbackOf(r), byDev = (m.bench && m.bench.byDevice) || [];
    Array.prototype.forEach.call(r.sel.options, function (o) {
      // The saved option resolves to the device the model loaded on (auto, fallback).
      if (o.defaultSelected && m.thresholdKey) o.dataset.key = m.thresholdKey;
      var k = o.dataset.key || "", p = lookup(m.deviceThresholds, k);
      o.dataset.thr = p === null ? "" : Number(p).toFixed(2);
      o.dataset.src = p === null ? fb.src + " " + fb.value : "profile";
      var b = k && byDev.find(function (d) { return d.device === k || devId(d.device) === devId(k); });
      o.dataset.bench = b && b.best !== null && b.best !== undefined ? Number(b.best).toFixed(2) : "";
    });
    if (r.def) {
      if (force) { U.set(r.def, fmtThr(m.threshold)); U.note(r.def, null); r.def.classList.remove("edited"); }
      else U.refresh(r.def, fmtThr(m.threshold), show);
      r.def.placeholder = j.globalThreshold.toFixed(2);
    }
    refreshProfiles(tr, m, force);
    // The field: rebase on the selected device, then refresh it to the server's value there
    // (an unsaved edit is kept).
    var wasEdited = U.edited(r.input), before = r.input.value;
    applyDevice(tr, false, true);
    var key = r.key.value, v = key ? lookup(m.deviceThresholds, key) : m.threshold;
    v = v === null || v === undefined ? "" : String(v);
    if (force || !wasEdited) {
      U.set(r.input, v); U.note(r.input, null); r.input.classList.remove("edited");
      if (!force && !U.same(r.input, before, v)) U.flash(r.input);
    } else U.refresh(r.input, v, function (x) { return x === "" ? "none (" + fb.value + " " + fb.src + ")" : x; });
    thrHint(tr, false);
  }
  card.addEventListener("change", function (e) {
    var t = e.target, tr = t.closest && t.closest("tr[data-model]");
    if (!tr) return;
    if (t.matches("td.model-device select")) applyDevice(tr, true, false);
    else if (t.matches('input[name^="profile_remove."]')) {
      var r = thrParts(tr);
      if (t.value === r.key.value) { r.input.value = t.checked ? "" : r.input.defaultValue; r.input.classList.toggle("edited", U.edited(r.input)); thrHint(tr, false); }
    }
  });
  card.addEventListener("input", function (e) {
    var t = e.target, tr = t.closest && t.closest("tr[data-model]");
    if (!tr || !t.matches("input.thr-input")) return;
    typed[tr.dataset.model + "\n" + thrParts(tr).key.value] = t.value;
  });

  // Models card: rows keyed by model; a new or removed model re-renders the card.
  function refreshCard(j, force) {
    var rows = card.querySelectorAll("tbody tr[data-model]");
    var names = Array.prototype.map.call(rows, function (r) { return r.dataset.model; }).join("\n");
    if (names !== j.models.map(function (m) { return m.key; }).join("\n")) { reloadCard(force); return; }
    var upd = function (ctl, v) {
      if (force) { U.set(ctl, v); U.note(ctl, null); ctl.classList.remove("edited"); return; }
      U.refresh(ctl, v, show);
    };
    j.models.forEach(function (m, i) {
      var tr = rows[i];
      upd(tr.querySelector("input[name=enabled]"), m.enabled);
      var rm = tr.querySelector("button.model-remove");
      if (rm) { rm.disabled = m.enabled; rm.title = m.enabled ? "Enabled: disable it first (untick Load and save), then remove it" : "Remove this entry from the config; its files stay"; }
      var sel = tr.querySelector("td.model-device select");
      upd(sel, m.device || "");
      refreshThreshold(tr, m, j, force);
      var run = tr.querySelector("td.running"); run.className = "running st-" + m.running.stateClass;
      // "Failed" and its reason on one truncated line (opens to the full text; kept open across refreshes).
      var fail = m.running.stateClass === "fail" && m.running.detail ? m.running.detail : null, old = run.querySelector("details.run-detail");
      var keepOpen = !!(old && old.open && fail && old.querySelector("summary").textContent === fail);
      run.replaceChildren(el("span", "run-text", m.running.text));
      if (fail) { var dt = el("details", "run-detail"), sm = el("summary", null, fail); sm.title = fail; dt.appendChild(sm); dt.open = keepOpen; run.appendChild(dt); }
      var badge = tr.querySelector(".pending-badge"); badge.hidden = !m.restartNeeded; badge.title = m.reasons.join("; ");
      tr.classList.toggle("pending", m.restartNeeded);
      var bh = tr.querySelector(".bench-hint"); bh.replaceChildren();
      if (m.bench) {
        var rec = m.bench.recommended, used = rec && m.running.device === rec;
        var a = el("a", null, rec ? (m.bench.inUse ? "benchmark: " + rec + (used ? " (in use)" : " (configured; restart to use)") : "benchmark recommends " + rec) : "benchmark: no recommendation");
        a.href = m.bench.href; a.title = m.bench.recommendation; bh.append(el("br"), a);
      }
    });
    // Default radio group (one control).
    var radios = card.querySelectorAll("input[name=default_model]"), def = j.models.find(function (m) { return m.default; });
    var groupEdited = Array.prototype.some.call(radios, function (r) { return r.checked !== r.defaultChecked; });
    var defName = def ? def.name : null;
    if (force || !groupEdited) {
      var changed = false;
      radios.forEach(function (r) { var on = r.value === defName; if (r.checked !== on) changed = true; r.checked = on; r.defaultChecked = on; });
      if (changed && !force) radios.forEach(function (r) { if (r.checked) U.flash(r); });
    } else {
      var mine = Array.prototype.find.call(radios, function (r) { return r.checked; });
      if (mine && mine.value === defName) radios.forEach(function (r) { r.defaultChecked = r.checked; });
    }
    card.querySelector(".restart-now").hidden = !j.restartNeeded;
  }

  // Re-render the Models card from a fresh /config, carrying the user's unsaved edits over.
  function reloadCard(force) {
    fetch("/config", {cache: "no-store"}).then(function (r) { return r.text(); }).then(function (html) {
      var doc = new DOMParser().parseFromString(html, "text/html"), fresh = doc.getElementById("models");
      if (!fresh) return;
      var oldRows = {};
      card.querySelectorAll("tbody tr[data-model]").forEach(function (r) { oldRows[r.dataset.model] = r; });
      if (!force) card.querySelectorAll("[data-key]").forEach(function (c) {
        if (!U.edited(c)) return;
        var k = c.dataset.key, target = c.type === "radio" ? fresh.querySelector('[data-key="default"][value="' + CSS.escape(c.value) + '"]') : fresh.querySelector('[data-key="' + CSS.escape(k) + '"]');
        if (!target) return;
        if (c.type === "checkbox" || c.type === "radio") target.checked = c.checked; else target.value = c.value;
        target.classList.add("edited");
      });
      var st = L.view.capture(card);
      card.replaceChildren.apply(card, Array.prototype.slice.call(document.adoptNode(fresh).childNodes));
      L.view.restore(card, st);
      card.querySelectorAll("tbody tr[data-model]").forEach(function (r) {
        if (!oldRows[r.dataset.model]) U.flash(r);
        // A carried-over device choice: point the threshold field at that device again.
        var sel = r.querySelector("td.model-device select");
        if (sel && U.edited(sel)) applyDevice(r, false, true);
      });
    }).catch(function () {});
  }

  function sourceLabel(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
  function banner(j, since) {
    var news = (j.changes || []).filter(function (c) { return c.revision > since && !own[c.revision]; });
    if (!news.length) return;
    var box = $("live-banner"), list = box.querySelector("ul");
    if (!list || box.hidden) { box.replaceChildren(); box.append(el("strong", null, "Changed elsewhere \u2014 the page now shows the new values (your unsaved edits are kept and marked):")); list = el("ul"); box.append(list);
      var x = el("button", "small", "Dismiss"); x.type = "button"; x.addEventListener("click", function () { box.hidden = true; }); box.append(x); }
    news.forEach(function (c) {
      list.append(el("li", null, sourceLabel(c.source) + ": " + (c.summary || "no visible change") + " (revision " + c.revision + ", " + new Date(c.time).toLocaleTimeString() + ")"));
      (c.fields || []).forEach(function (f) {
        var m = /^model:[^:]*:(.*)$/.exec(f.key);
        var row = m && card.querySelector('tr[data-model="' + CSS.escape(m[1]) + '"]');
        if (row) U.flash(row);
      });
    });
    box.hidden = false;
  }

  function pendingBox(j) {
    var box = $("pending-box"), ul = $("pending-list");
    ul.replaceChildren(); (j.pending.all || []).forEach(function (p) { ul.append(el("li", null, p)); });
    box.hidden = !j.restartNeeded;
  }

  L.onConfig(function (j) {
    var reset = resetAfter && j.revision >= resetAfter.rev ? resetAfter.form : null;
    if (epoch !== null && j.epoch !== epoch) pageRev = 0;
    epoch = j.epoch;
    if (j.revision !== pageRev || reset) {
      refreshServer(j, reset === serverForm || reset === jsonForm ? reset : null);
      refreshCard(j, reset === card);
      banner(j, pageRev);
      pageRev = j.revision; $("cfg-rev").textContent = j.revision;
      if (reset) resetAfter = null;
    } else {
      // Same config: only the running state may have changed.
      refreshCard(j, false);
    }
    pendingBox(j);
  });

  // "use" next to the benchmark's best threshold.
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("[data-use-threshold]");
    if (!b) return;
    var tr = b.closest("tr"), input = tr.querySelector("input.thr-input");
    input.value = b.dataset.useThreshold; input.classList.toggle("edited", U.edited(input)); U.flash(input);
    if (tr.dataset.model) typed[tr.dataset.model + "\n" + thrParts(tr).key.value] = input.value;
  });

  // Save without leaving the page: JSON answers shown next to the button.
  function msgOf(form) { return form.querySelector(".form-msg"); }
  function say(form, cls, text) { var m = msgOf(form); m.className = "form-msg " + cls; m.textContent = text; }
  // `by`: the clicked button, disabled while the request runs. `opts.url`: post there instead
  // of the form's action; `opts.keep`: the form keeps its unsaved edits (the answer is still
  // shown in it).
  function send(form, data, by, opts) {
    opts = opts || {};
    say(form, "muted", opts.url ? "working…" : "saving…");
    if (by) by.disabled = true;
    var done = function () { if (by) by.disabled = false; };
    return fetch(opts.url || form.getAttribute("action").split("#")[0], {method: "POST", body: new URLSearchParams(data), headers: {"Accept": "application/json"}})
      .then(function (r) { return r.json().then(function (j) { j._status = r.status; return j; }); })
      .then(function (j) {
        done();
        if (!j.success) { say(form, "error-text", j.message); return; }
        own[j.revision] = true;
        say(form, "ok-text", j.message);
        if (j.restarting) { say(form, "ok-text", j.message + " Restarting…"); L.restarting(); L.afterRestart(); return; }
        if (!opts.keep) resetAfter = {form: form, rev: j.revision};
        L.poll(true);
      })
      .catch(function (e) { done(); say(form, "error-text", "Not saved: the server did not answer (" + e + "); it may be restarting or stopped, so check that it runs and save again."); });
  }
  // Remove on a Models card row: its button belongs to the empty #model-remove form (so it is
  // not the card's default button); only `name` is sent, the answer shows in the card, the
  // other rows keep their unsaved edits (the card re-renders as the model list changed).
  var removeForm = $("model-remove");
  document.addEventListener("submit", function (e) {
    if (!removeForm || e.target !== removeForm) return;
    var by = e.submitter;
    e.preventDefault();
    if (!by || !by.name) return;
    var data = new FormData();
    data.append(by.name, by.value);
    send(card, data, by, {url: removeForm.getAttribute("action").split("#")[0], keep: true});
  });
  document.addEventListener("submit", function (e) {
    var form = e.target;
    if (form !== serverForm && form !== card && form !== jsonForm) return;
    var by = e.submitter;
    if (by && by.hasAttribute("data-restart")) { e.preventDefault(); L.restart(by); return; }
    e.preventDefault();
    var data = new FormData(form);
    if (by && by.name) data.append(by.name, by.value);
    send(form, data, by);
  });
})();
// Resources: groups are <details> (open when a row needs attention); a filter box hides the
// rows that do not match and opens the groups with hits; the export group's one "Add exported
// models to the config" box sets every export form's add_to_config on submit. The section is
// re-rendered in place (above), so the filter text and the box are kept here and re-applied.
(function () {
  var box = document.getElementById("resources");
  if (!box) return;
  var query = "", addExport = true;
  function groups() { return box.querySelectorAll("details.res-group"); }
  function apply() {
    var terms = query.toLowerCase().split(/\s+/).filter(Boolean), total = 0;
    groups().forEach(function (g) {
      var hits = 0;
      g.querySelectorAll("table.resources tbody tr").forEach(function (tr) {
        var text = tr.textContent.toLowerCase();
        var ok = terms.every(function (t) { return text.indexOf(t) >= 0; });
        tr.hidden = !ok; if (ok) hits++;
      });
      total += hits;
      if (terms.length) {
        if (g.dataset.wasOpen === undefined) g.dataset.wasOpen = g.open ? "1" : "";
        g.hidden = !hits; if (hits) g.open = true;
      } else {
        g.hidden = false;
        if (g.dataset.wasOpen !== undefined) { g.open = !!g.dataset.wasOpen; delete g.dataset.wasOpen; }
      }
    });
    var count = document.getElementById("res-filter-count"), none = document.getElementById("res-filter-none");
    if (count) count.textContent = terms.length ? total + (total === 1 ? " match" : " matches") : "";
    if (none) none.hidden = !terms.length || total > 0;
  }
  function wire() {
    var f = document.getElementById("res-filter"), cb = document.getElementById("export-add-to-config");
    if (f) { f.closest(".res-filter-row").hidden = false; if (f.value !== query) f.value = query; }
    if (cb) { cb.closest(".export-add-row").hidden = false; cb.checked = addExport; }
    if (query) apply();
  }
  box.addEventListener("input", function (e) {
    if (e.target && e.target.id === "res-filter") { query = e.target.value; apply(); }
  });
  box.addEventListener("change", function (e) {
    if (e.target && e.target.id === "export-add-to-config") addExport = e.target.checked;
  });
  // Capture phase: before the data-ajax handler builds the request from the form.
  document.addEventListener("submit", function (e) {
    var form = e.target;
    if (!form.matches || !form.matches("form.export")) return;
    var add = form.querySelector("input.export-add");
    if (add) add.value = addExport ? "1" : "0";
  }, true);
  if (window.MutationObserver) new MutationObserver(function (list) {
    if (list.some(function (m) { return m.target === box; })) wire();
  }).observe(box, {childList: true});
  // A link or #fragment to a row or group inside a closed group opens it.
  function reveal(hash) {
    var id = hash && hash.charAt(0) === "#" ? decodeURIComponent(hash.slice(1)) : "";
    var t = id && document.getElementById(id);
    for (var d = t && t.closest("details"); d; d = d.parentElement && d.parentElement.closest("details")) d.open = true;
  }
  document.addEventListener("click", function (e) {
    var a = e.target.closest && e.target.closest('a[href^="#"]');
    if (a) reveal(a.getAttribute("href"));
  });
  window.addEventListener("hashchange", function () { reveal(location.hash); });
  reveal(location.hash);
  wire();
})();
