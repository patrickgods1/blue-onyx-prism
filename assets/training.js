// Training page: the forms' show / hide (import source, auto-label mode, night augmentation,
// the model name with several models ticked; without JS every field shows and works), live
// progress of the queued and running training runs and import / auto-label jobs (polls
// GET /v1/training) and a reload when one finishes, so the results show.
// Auto-label after import: a ticked import posts the Auto-label card's fields as `al.<name>`
// hidden inputs (its classes included); the box and its follow-up options are remembered in
// localStorage. A Recent imports row's Auto-label button fills the card's filter (import=<id>).
(function () {
  "use strict";

  // Set by the auto-label classes block: the picker's selection, whether some are outside the
  // class list, and the confirm that appends them (null without the picker).
  var alClasses = null;

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
  // Auto-label: the VLM check's options only while it is on (and the minimum score only while
  // ticked); the form's active mode for the CSS (labeler badges hide in model mode). Runs after
  // the mode toggle above, which re-enables every field of the shown mode.
  (function () {
    var form = document.getElementById("autolabel-form");
    if (!form) return;
    var vlm = form.elements.vlm, minOn = form.elements.vlm_min_on, min = form.elements.vlm_min_score;
    var opts = document.getElementById("al-vlm-options");
    var update = function () {
      form.dataset.active = checked(form, "mode");
      if (!vlm || vlm.disabled) return;
      var off = vlm.value === "off";
      if (opts) {
        opts.classList.toggle("muted", off);
        opts.querySelectorAll("input").forEach(function (i) { i.disabled = off; });
      }
      if (min && minOn) min.disabled = off || !minOn.checked;
    };
    [vlm, minOn].forEach(function (e) { if (e) e.addEventListener("change", update); });
    form.querySelectorAll('input[name="mode"]').forEach(function (r) { r.addEventListener("change", update); });
    update();
  })();

  // Auto-label classes: the class picker (class-picker.js). The chosen classes post as repeated
  // hidden `classes` fields; one outside the class list is appended to it first (after a confirm,
  // POST /v1/review/classes, append only).
  (function () {
    var form = document.getElementById("autolabel-form"), CP = window.ClassPicker, MP = window.ModelPicker;
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
    // Model mode: the model picker (model-picker.js) posts with_model / device / the confirms; the
    // preset sets the status filter; the pre-accept score must reach the threshold.
    var mpHolder = document.getElementById("al-model-picker");
    var mp = mpHolder && MP ? MP.create(mpHolder, {
      onLoad: function (j) {
        var t = form.elements.threshold;
        if (t && j.default_threshold && t.value === t.defaultValue) t.value = j.default_threshold;
      }
    }) : null;
    var preset = document.getElementById("al-preset");
    if (preset && form.elements.status) {
      preset.addEventListener("change", function () {
        form.elements.status.value = preset.value === "self-training" ? "any" : "unreviewed";
      });
    }
    var modelProblemOk = function (ev) {
      if (!mp) return true;
      var p = mp.problem() || MP.util.acceptProblem(form.elements.threshold.value, form.elements.model_accept_score.value);
      if (!p) return true;
      ev.preventDefault();
      mp.say(p, true);
      return false;
    };
    var mode = function () { var r = form.querySelector('input[name="mode"]:checked'); return r ? r.value : ""; };
    var fields = function (names) {
      form.querySelectorAll("input.cp-field").forEach(function (i) { i.remove(); });
      names.forEach(function (n) {
        var i = document.createElement("input");
        i.type = "hidden"; i.name = "classes"; i.value = n; i.className = "cp-field";
        form.appendChild(i);
      });
    };
    alClasses = function () {
      if (!picker) return null;
      var names = picker.getSelected();
      return {
        names: names,
        missing: CP.util.missingFrom(names, catalog.classes || []).length > 0,
        ensure: function () { return CP.ensureInList(names, catalog); }
      };
    };
    form.addEventListener("submit", function (ev) {
      if (mode() === "model" && !modelProblemOk(ev)) return;
      if (!picker) { fields([]); return; }
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

  // Auto-label after import: the follow-up options show while the box is ticked (hidden ones
  // are disabled, so not posted), the Review folder switch while "Save frames to" has a value.
  (function () {
    var form = document.getElementById("import-form");
    var alForm = document.getElementById("autolabel-form");
    var box = document.getElementById("import-autolabel");
    if (!form || !box) return;
    var after = document.getElementById("import-after");
    var folder = document.getElementById("import-after-folder");
    var out = form.elements.out_dir;
    var msg = document.getElementById("import-autolabel-msg");
    var KEY = "bop.training.importAfter";
    var NAMES = ["autolabel", "after_reject", "after_accept", "after_review_folder"];
    var say = function (t) { if (msg) { msg.textContent = t || ""; msg.className = "form-msg" + (t ? " error-text" : ""); } };
    var show = function (el, on) {
      if (!el) return;
      el.hidden = !on;
      el.querySelectorAll("input").forEach(function (i) { i.disabled = !on; });
    };
    var update = function () {
      show(after, box.checked);
      show(folder, !!(out && out.value.trim()));
    };
    var remember = function () {
      var v = {};
      NAMES.forEach(function (n) { var e = form.elements[n]; if (e) v[n] = !!e.checked; });
      var sc = form.elements.after_reject_score;
      if (sc) v.after_reject_score = sc.value;
      try { window.localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) { /* private mode */ }
    };
    (function restore() {
      var v = null;
      try { v = JSON.parse(window.localStorage.getItem(KEY) || "null"); } catch (e) { v = null; }
      if (!v || typeof v !== "object") return;
      NAMES.forEach(function (n) { var e = form.elements[n]; if (e && typeof v[n] === "boolean") e.checked = v[n]; });
      var sc = form.elements.after_reject_score;
      if (sc && typeof v.after_reject_score === "string" && v.after_reject_score) sc.value = v.after_reject_score;
    })();
    NAMES.concat(["after_reject_score"]).forEach(function (n) {
      var e = form.elements[n];
      if (e) e.addEventListener("change", function () { remember(); update(); });
    });
    if (out) { out.addEventListener("input", update); out.addEventListener("change", update); }
    update();

    var confirmed = false;
    form.addEventListener("submit", function (ev) {
      form.querySelectorAll("input.al-injected").forEach(function (i) { i.remove(); });
      say("");
      remember();
      if (!box.checked || !alForm) return;
      var cls = alClasses ? alClasses() : null;
      if (cls && !cls.names.length) {
        ev.preventDefault();
        say("Auto-label: choose at least one class to look for on the Auto-label card.");
        return;
      }
      if (cls && cls.missing && !confirmed) {
        ev.preventDefault();
        var by = ev.submitter;
        cls.ensure().then(function () {
          confirmed = true;
          if (form.requestSubmit) form.requestSubmit(by || undefined); else form.submit();
        }).catch(function (e) { say(e.cancelled ? "Not imported: the class list was left as it is." : e.message); });
        return;
      }
      confirmed = false;
      var add = function (name, value) {
        var i = document.createElement("input");
        i.type = "hidden"; i.name = "al." + name; i.value = value; i.className = "al-injected";
        form.appendChild(i);
      };
      new FormData(alForm).forEach(function (value, name) {
        if (typeof value !== "string") return;
        if (cls && name === "classes") return;
        add(name, value);
      });
      if (cls) cls.names.forEach(function (n) { add("classes", n); });
    });

    // Recent imports: Auto-label fills the card's filter with the import and scrolls to it.
    document.querySelectorAll("[data-al-fallback]").forEach(function (a) { a.hidden = true; });
    document.querySelectorAll("button[data-al-import]").forEach(function (b) {
      b.hidden = false;
      b.addEventListener("click", function () {
        if (!alForm) return;
        var frames = document.getElementById("al-frames");
        var status = document.getElementById("al-status");
        var limit = document.getElementById("al-limit");
        var n = parseInt(b.getAttribute("data-al-frames"), 10);
        if (!(n >= 1)) n = 1;
        if (frames) {
          frames.value = "import=" + b.getAttribute("data-al-import");
          var d = frames.closest("details");
          if (d) d.open = true;
        }
        if (status) status.value = "any";
        if (limit) limit.value = String(Math.min(n, 20000));
        var card = document.getElementById("autolabel");
        if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
        var submit = alForm.querySelector('.actions button[type="submit"]');
        if (submit) submit.focus({ preventScroll: true });
      });
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
  var isBusy = function (s) { return s === "queued" || s === "running" || s === "paused"; };
  [[runs, "run"], [tasks, "task"]].forEach(function (p) {
    if (!p[0]) return;
    p[0].querySelectorAll("tr[data-" + p[1] + "]").forEach(function (tr) {
      var st = tr.querySelector('[data-field="state"]');
      if (st && isBusy(st.textContent)) busy[tr.dataset[p[1]]] = true;
    });
  });

  // Metrics (curves, per-epoch table, columns) of the runs that report a history: rendered on
  // load for every run (finished ones too) and again when the history, batch or columns change.
  var metricsState = { columns: [], metrics: [], data: null };
  function logBoxes() {
    // The Log <details> keeps its open state per run for this tab.
    document.querySelectorAll("details.tr-logbox").forEach(function (d) {
      var key = "prism.training.log." + d.dataset.run;
      try { if (window.sessionStorage.getItem(key) === "1") d.open = true; } catch (e) { /* storage blocked */ }
      d.addEventListener("toggle", function () {
        try { window.sessionStorage.setItem(key, d.open ? "1" : "0"); } catch (e) { /* storage blocked */ }
      });
    });
  }
  logBoxes();
  function availableKeys(data) {
    var seen = {}, raw = [];
    (data.runs || []).forEach(function (r) {
      var h = r.progress && r.progress.history;
      if (h) window.TrainMetrics.keysOf(h).forEach(function (k) { if (!seen[k]) { seen[k] = true; raw.push(k); } });
    });
    var known = (data.metrics || []).map(function (m) { return m.id; }).filter(function (id) { return seen[id]; });
    return known.concat(raw.filter(function (k) { return known.indexOf(k) < 0; }).sort());
  }
  function saveColumns(list) {
    return fetch("/v1/training/metric-columns", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ columns: list })
    }).then(function (res) {
      return res.json().catch(function () { return null; }).then(function (j) {
        if (!res.ok || !j || j.success === false) {
          return { ok: false, message: (j && j.message) || "The server answered HTTP " + res.status + " (the columns were not saved)." };
        }
        metricsState.columns = j.columns || list;
        // renderMetrics takes the columns from the last poll's answer: keep it current.
        if (metricsState.data) metricsState.data.metric_columns = metricsState.columns;
        renderMetrics(metricsState.data, true);
        return { ok: true };
      });
    });
  }
  function renderMetrics(data, force) {
    if (!data || !window.TrainMetrics) return;
    metricsState.data = data;
    if (data.metric_columns) metricsState.columns = data.metric_columns;
    if (data.metrics) metricsState.metrics = data.metrics;
    var avail = availableKeys(data);
    var byId = {};
    (data.runs || []).forEach(function (r) { byId[r.id] = r; });
    document.querySelectorAll(".tr-metrics[data-run]").forEach(function (host) {
      var r = byId[host.dataset.run];
      var sig = r ? window.TrainMetrics.signature(r.progress, metricsState.columns) + "|" + r.state + "|" + avail.length : "";
      var row = host.closest("tr");
      if (!force && host.dataset.sig === sig) return;
      host.dataset.sig = sig;
      if (r && r.progress) {
        window.TrainMetrics.render(host, r.progress, metricsState.columns, metricsState.metrics, {
          runId: r.id, available: avail, running: r.state === "running", onColumns: saveColumns
        });
      } else {
        host.replaceChildren();
      }
      if (row && row.classList.contains("tr-metrics-row")) row.hidden = host.childNodes.length === 0;
    });
  }
  function loadMetrics() {
    if (!window.TrainMetrics || !document.querySelector(".tr-metrics[data-run]")) return;
    fetch("/v1/training", { headers: { Accept: "application/json" } })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) { if (data) renderMetrics(data, false); })
      .catch(function () {});
  }
  loadMetrics();

  if (Object.keys(busy).length === 0) return;
  function describe(r, list) {
    if (r.state === "queued") {
      if (!r.after) return "queued";
      var dep = (list || []).filter(function (x) { return x.id === r.after; })[0];
      return "queued (after " + (dep && dep.kind === "import" ? "import " : "") + r.after + ")";
    }
    if (r.state !== "running" && r.state !== "paused") return r.state;
    var s = r.stage ? String(r.stage).replace(/_/g, " ") : "starting";
    if (r.state === "paused") return r.stage ? "paused (" + s + " " + r.step + "/" + r.steps + ")" + (r.detail ? ": " + r.detail : "") : "paused";
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
      set("describe", describe(r, list));
      if (r.log && r.log.length) {
        var logEl = tr.querySelector('[data-field="log"]');
        var box = logEl && logEl.closest("details.tr-logbox");
        if (box) { box.hidden = false; set("log", r.log.slice(-40).join("\n")); }
        else set("log", r.log.slice(-6).join("\n"));
      }
      // Pause while running, Resume while paused (Cancel in both).
      var pause = tr.querySelector('[data-field="pause"]');
      var resume = tr.querySelector('[data-field="resume"]');
      if (pause) pause.hidden = r.state !== "running";
      if (resume) resume.hidden = r.state !== "paused";
      if (busy[r.id] && !isBusy(r.state)) finished = true;
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
        renderMetrics(data, false);
        // Not reload(): the page may be the answer to a POST, and reloading it would ask to
        // resubmit (and queue the job again).
        if (a || b) window.location.assign("/training" + window.location.hash);
      })
      .catch(function () {});
  }
  setInterval(tick, 3000);
})();
