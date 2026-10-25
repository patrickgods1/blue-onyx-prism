// Training page: the forms' show / hide (import source, auto-label mode, night augmentation,
// the model name with several models ticked; without JS every field shows and works), live
// progress of the queued and running training runs and import / auto-label jobs (polls
// GET /v1/training) and a reload when one finishes, so the results show.
(function () {
  "use strict";

  // The value of the checked radio `name` of `form`.
  function checked(form, name) {
    var r = form.querySelector('input[name="' + name + '"]:checked');
    return r ? r.value : "";
  }

  // Show the `[attr]` blocks of `form` whose space-separated list names `value`. A hidden
  // block's fields are disabled (not posted, not validated: a stale value there must not block
  // the form); the server ignores the fields the choice does not use anyway.
  function toggle(form, name, attr) {
    if (!form) return;
    var update = function () {
      var v = checked(form, name);
      form.querySelectorAll("[" + attr + "]").forEach(function (el) {
        var off = el.getAttribute(attr).split(" ").indexOf(v) < 0;
        el.hidden = off;
        el.querySelectorAll("input, select, textarea").forEach(function (f) { f.disabled = off; });
      });
    };
    form.querySelectorAll('input[name="' + name + '"]').forEach(function (r) {
      r.addEventListener("change", update);
    });
    update();
  }
  toggle(document.getElementById("import-form"), "source", "data-source");

  // An invalid field inside a closed <details> (Sampling options, Advanced) cannot show its
  // message: open it first.
  document.querySelectorAll("form").forEach(function (f) {
    f.addEventListener("invalid", function (ev) {
      var d = ev.target.closest && ev.target.closest("details");
      if (d && !d.open) d.open = true;
    }, true);
  });

  // Enter in the Train form's "Another folder" saves the folder (the form's default button
  // queues training).
  var trainFolder = document.getElementById("train-folder");
  if (trainFolder && trainFolder.form) {
    trainFolder.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter") return;
      var save = trainFolder.form.querySelector('button[name="save"]');
      if (save) { ev.preventDefault(); save.click(); }
    });
  }

  // The import preview: what the path holds and the per-video budget, one line under the
  // field (GET /v1/training/import/preview), debounced; Browse fills the field and fires
  // `input`, so a picked path is previewed too.
  var importForm = document.getElementById("import-form");
  var preview = document.getElementById("import-preview");
  if (importForm && preview) {
    var pathField = importForm.elements.path;
    var timer = null;
    var seq = 0;
    var show = function (text, warn) {
      preview.textContent = text;
      preview.hidden = !text;
      preview.classList.toggle("error-text", !!warn);
    };
    var run = function () {
      var p = pathField ? pathField.value.trim() : "";
      var mine = ++seq;
      if (!p) { show("", false); return; }
      var q = "path=" + encodeURIComponent(p);
      var mc = importForm.elements.max_candidates;
      if (mc && mc.value.trim()) q += "&max_candidates=" + encodeURIComponent(mc.value.trim());
      var cam = importForm.elements.camera;
      if (cam && cam.value.trim()) q += "&camera=" + encodeURIComponent(cam.value.trim());
      show("Checking\u2026", false);
      fetch("/v1/training/import/preview?" + q, { headers: { Accept: "application/json" } })
        .then(function (res) { return res.json(); })
        .then(function (j) {
          if (mine !== seq) return;
          if (!j.success) { show(j.message || "Cannot read this path.", true); return; }
          var w = (j.warnings || []).filter(function (x) { return x.indexOf(".bvr") < 0; });
          show(j.summary + (w.length ? ". " + w.join(". ") : ""), false);
        })
        .catch(function () { if (mine === seq) show("", false); });
    };
    var later = function () {
      if (timer) clearTimeout(timer);
      timer = setTimeout(run, 400);
    };
    ["path", "max_candidates", "camera"].forEach(function (n) {
      var el = importForm.elements[n];
      if (el) { el.addEventListener("input", later); el.addEventListener("change", later); }
    });
    if (pathField && pathField.value.trim()) run();
  }
  toggle(document.getElementById("autolabel-form"), "mode", "data-mode");

  // Auto-label classes: the class picker (class-picker.js). The chosen classes post as repeated
  // hidden `classes` fields; one outside the class list is appended to it first (after a confirm,
  // POST /v1/review/classes, append only).
  (function () {
    var form = document.getElementById("autolabel-form"), CP = window.ClassPicker;
    var holder = document.getElementById("al-picker"), msg = document.getElementById("al-classes-msg");
    if (!form || !holder || !CP) return;
    var picker = null, catalog = null, confirmed = false;
    var say = function (t, bad) { if (msg) { msg.textContent = t || ""; msg.className = "form-msg" + (bad ? " error-text" : ""); } };
    var entriesOf = function (cat) {
      var out = CP.util.mergeEntries(cat.entries || [], (cat.classes || []));
      out.forEach(function (e) { if ((cat.classes || []).indexOf(e.name) >= 0) e.in_list = true; e.hidden = (cat.hidden || []).indexOf(e.name) >= 0; });
      return out;
    };
    CP.fetchCatalog().then(function (cat) {
      catalog = cat;
      picker = CP.create(holder, {
        entries: entriesOf(cat), selected: CP.util.defaultSelection(cat), mode: "full", allowAdd: true,
        quickSets: CP.util.quickSetsFor(cat)
      });
    }).catch(function (e) {
      holder.textContent = "";
      var p = document.createElement("p");
      p.className = "muted small";
      p.textContent = "The class list could not be read (" + e.message + "); the labelers use the class list and training.synonyms.";
      holder.appendChild(p);
    });
    var mode = function () { var r = form.querySelector('input[name="mode"]:checked'); return r ? r.value : ""; };
    var fields = function (names) {
      form.querySelectorAll("input.cp-field").forEach(function (i) { i.remove(); });
      names.forEach(function (n) {
        var i = document.createElement("input");
        i.type = "hidden"; i.name = "classes"; i.value = n; i.className = "cp-field";
        form.appendChild(i);
      });
    };
    form.addEventListener("submit", function (ev) {
      if (!picker || mode() !== "labelers") { fields([]); return; }
      var names = picker.getSelected();
      if (!names.length) { ev.preventDefault(); say("Choose at least one class to look for.", true); return; }
      say("", false);
      fields(names);
      if (confirmed || !CP.util.missingFrom(names, catalog.classes || []).length) { confirmed = false; return; }
      ev.preventDefault();
      var by = ev.submitter;
      CP.ensureInList(names, catalog).then(function () {
        confirmed = true;
        if (form.requestSubmit) form.requestSubmit(by || undefined); else form.submit();
      }).catch(function (e) { say(e.cancelled ? "Not queued: the class list was left as it is." : e.message, true); });
    });
  })();

  var train = document.getElementById("train");
  if (train) {
    // The model name: one ticked model only (the server refuses it with more).
    var name = document.getElementById("train-name");
    var hint = document.getElementById("train-name-hint");
    var models = train.querySelectorAll('input[name="model"]');
    var hintText = hint ? hint.innerHTML : "";
    var names = function () {
      var n = 0;
      models.forEach(function (m) { if (m.checked) n++; });
      if (!name) return;
      name.disabled = n > 1;
      if (hint) {
        if (n > 1) hint.textContent = n + " models ticked: each gets its own name (night-/custom-<model>-<date>).";
        else hint.innerHTML = hintText;
      }
    };
    models.forEach(function (m) { m.addEventListener("change", names); });
    names();

    // Night augmentation: its probabilities show (and post) only when ticked; prefilled with
    // the server's defaults when empty.
    var night = document.getElementById("train-night");
    var fields = document.getElementById("train-night-fields");
    if (night && fields) {
      var defaults = { aug_dark: "0.5", aug_noise: "0.3", aug_gray: "0.5" };
      var aug = function (prefill) {
        fields.hidden = !night.checked;
        fields.querySelectorAll("input").forEach(function (i) {
          i.disabled = !night.checked;
          if (prefill && night.checked && i.value === "" && defaults[i.name]) i.value = defaults[i.name];
        });
      };
      night.addEventListener("change", function () { aug(true); });
      aug(false);
    }
  }

  // Live progress.
  var runs = document.getElementById("training-runs");
  var tasks = document.getElementById("training-tasks");
  var busy = {};
  [[runs, "run"], [tasks, "task"]].forEach(function (p) {
    if (!p[0]) return;
    p[0].querySelectorAll("tr[data-" + p[1] + "]").forEach(function (tr) {
      var st = tr.querySelector('[data-field="state"]');
      if (st && (st.textContent === "queued" || st.textContent === "running")) busy[tr.dataset[p[1]]] = true;
    });
  });
  if (Object.keys(busy).length === 0) return;
  function describe(r) {
    if (r.state === "queued") return "queued";
    if (r.state !== "running") return r.state;
    var s = r.stage ? String(r.stage).replace(/_/g, " ") : "starting";
    return s + " (" + r.step + "/" + r.steps + ")" + (r.detail ? ": " + r.detail : "");
  }
  function update(table, attr, list) {
    var finished = false;
    if (!table) return false;
    (list || []).forEach(function (r) {
      var tr = table.querySelector("tr[data-" + attr + '="' + r.id + '"]');
      if (!tr) return;
      var set = function (f, v) {
        var el = tr.querySelector('[data-field="' + f + '"]');
        if (!el) return;
        if (f === "percent") el.value = v; else el.textContent = v;
      };
      set("state", r.state);
      set("percent", r.percent);
      set("describe", describe(r));
      if (r.log && r.log.length) set("log", r.log.slice(-6).join("\n"));
      if (busy[r.id] && r.state !== "queued" && r.state !== "running") finished = true;
    });
    return finished;
  }
  function tick() {
    fetch("/v1/training", { headers: { Accept: "application/json" } })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data) return;
        var a = update(runs, "run", data.runs);
        var b = update(tasks, "task", data.tasks);
        // Not reload(): the page may be the answer to a POST, and reloading it would ask to
        // resubmit (and queue the job again).
        if (a || b) window.location.assign("/training" + window.location.hash);
      })
      .catch(function () {});
  }
  setInterval(tick, 3000);
})();
