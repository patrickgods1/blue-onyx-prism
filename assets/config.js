// Blue Onyx Prism: Config page (served from /static/; no inline scripts under the CSP).
// Server-side file browser for the Path / Classes inputs (GET /v1/fs/browse).
(function () {
  var dlg = document.getElementById("browse-dialog");
  if (!window.fetch || !dlg || typeof dlg.showModal !== "function") return;
  var $ = function (id) { return document.getElementById(id); };
  var list = $("browse-list"), pathEl = $("browse-path"), msg = $("browse-msg"), roots = $("browse-roots");
  var up = $("browse-up"), use = $("browse-use"), title = $("browse-title");
  var kind = "model", target = null, cur = null, seq = 0;
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  function pick(path) {
    target.value = path;
    target.dispatchEvent(new Event("input", { bubbles: true }));
    dlg.close();
    target.focus();
  }
  function render(j) {
    cur = j;
    pathEl.textContent = j.path;
    msg.textContent = j.truncated ? "Only the first entries are shown." : (j.entries.length ? "" : "Nothing to choose here.");
    up.disabled = !j.parent;
    up.dataset.path = j.parent || "";
    roots.textContent = "";
    var here = el("option", null, "Places\u2026"); here.value = ""; roots.appendChild(here);
    j.roots.forEach(function (r) { var o = el("option", null, r.label + "  " + r.path); o.value = r.path; roots.appendChild(o); });
    roots.value = "";
    use.hidden = kind !== "model";
    list.textContent = "";
    var sel = null;
    j.entries.forEach(function (e) {
      var b = el("button", "browse-item" + (e.dir ? " dir" : " file"));
      b.type = "button";
      b.appendChild(el("span", "browse-name", (e.dir ? "\uD83D\uDCC1 " : "") + e.name));
      b.appendChild(el("span", "browse-size", e.dir ? "" : e.sizeText));
      if (!e.dir && j.selected === e.name) { b.classList.add("selected"); sel = b; }
      b.addEventListener("click", function () { if (e.dir) go(e.path, false); else pick(e.path); });
      list.appendChild(b);
    });
    if (sel && sel.scrollIntoView) sel.scrollIntoView({ block: "nearest" });
  }
  function go(path, retry) {
    var n = ++seq;
    var url = "/v1/fs/browse?kind=" + encodeURIComponent(kind) + (path ? "&path=" + encodeURIComponent(path) : "");
    fetch(url, { headers: { Accept: "application/json" } }).then(function (r) {
      return r.json().then(function (j) { return { status: r.status, body: j }; });
    }).then(function (res) {
      if (n !== seq) return;
      if (res.status === 200) { render(res.body); if (retry) msg.textContent = retry; return; }
      var m = (res.body && res.body.message) || "Cannot list that folder.";
      if (res.status === 404 && path) { go("", m); return; }
      msg.textContent = m;
    }).catch(function () { if (n === seq) msg.textContent = "Cannot reach the server."; });
  }
  document.querySelectorAll("button.browse").forEach(function (b) {
    b.hidden = false;
    b.addEventListener("click", function () {
      kind = b.dataset.browse === "classes" ? "classes" : "model";
      target = b.form.elements[b.dataset.target];
      title.textContent = kind === "classes" ? "Choose a classes YAML" : "Choose a model file or folder";
      list.textContent = ""; msg.textContent = "Loading\u2026";
      dlg.showModal();
      go(target.value.trim(), "");
    });
  });
  roots.addEventListener("change", function () { if (roots.value) go(roots.value, false); });
  up.addEventListener("click", function () { if (up.dataset.path) go(up.dataset.path, false); });
  use.addEventListener("click", function () { if (cur) pick(cur.path); });
  $("browse-cancel").addEventListener("click", function () { dlg.close(); });
})();
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
      upd(tr.querySelector("select"), m.device || "");
      var thr = tr.querySelector("input[type=number]");
      upd(thr, fmtThr(m.threshold));
      thr.placeholder = j.globalThreshold.toFixed(2);
      var run = tr.querySelector("td.running"); run.textContent = m.running.text; run.className = "running st-" + m.running.stateClass;
      var badge = tr.querySelector(".pending-badge"); badge.hidden = !m.restartNeeded; badge.title = m.reasons.join("; ");
      tr.classList.toggle("pending", m.restartNeeded);
      var bh = tr.querySelector(".bench-hint"); bh.replaceChildren();
      if (m.bench) {
        var rec = m.bench.recommended, used = rec && m.running.device === rec;
        var a = el("a", null, rec ? (m.bench.inUse ? "benchmark: " + rec + (used ? " (in use)" : " (configured; restart to use)") : "benchmark recommends " + rec) : "benchmark: no recommendation");
        a.href = m.bench.href; a.title = m.bench.recommendation; bh.append(el("br"), a);
      }
      var th = tr.querySelector(".thr-hint"); th.replaceChildren();
      if (m.bench && m.bench.bestThreshold !== null && m.bench.bestThreshold !== undefined) {
        var t = m.bench.bestThreshold.toFixed(2);
        th.append(el("br"), el("span", "muted", "benchmark best " + t + " (" + m.bench.objective + ")"), " ");
        if (m.bench.thresholdInUse) th.append(el("span", "muted", m.running.threshold === m.bench.bestThreshold ? "in use" : "configured; restart to use"));
        else { var b = el("button", "small", "use"); b.type = "button"; b.dataset.useThreshold = t; b.title = "Put " + t + " in the threshold field (then save)"; th.append(b); }
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
      card.querySelectorAll("tbody tr[data-model]").forEach(function (r) { if (!oldRows[r.dataset.model]) U.flash(r); });
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
    var input = b.closest("tr").querySelector("input[type=number]");
    input.value = b.dataset.useThreshold; input.classList.toggle("edited", U.edited(input)); U.flash(input);
  });

  // Save without leaving the page: JSON answers shown next to the button.
  function msgOf(form) { return form.querySelector(".form-msg"); }
  function say(form, cls, text) { var m = msgOf(form); m.className = "form-msg " + cls; m.textContent = text; }
  // `by`: the clicked button, disabled while the request runs.
  function send(form, data, by) {
    say(form, "muted", "saving…");
    if (by) by.disabled = true;
    var done = function () { if (by) by.disabled = false; };
    return fetch(form.getAttribute("action").split("#")[0], {method: "POST", body: new URLSearchParams(data), headers: {"Accept": "application/json"}})
      .then(function (r) { return r.json().then(function (j) { j._status = r.status; return j; }); })
      .then(function (j) {
        done();
        if (!j.success) { say(form, "error-text", j.message); return; }
        own[j.revision] = true;
        say(form, "ok-text", j.message);
        if (j.restarting) { say(form, "ok-text", j.message + " Restarting…"); L.restarting(); L.afterRestart(); return; }
        resetAfter = {form: form, rev: j.revision};
        L.poll(true);
      })
      .catch(function (e) { done(); say(form, "error-text", "Not saved: " + e); });
  }
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
