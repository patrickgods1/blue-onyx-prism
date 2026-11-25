// Train area, every tab: the status strip under the tabs (counts per tab and the one suggested next
// action, GET /v1/train/summary, filled after the page is up so a large dataset never holds it back),
// the Jobs tab's list of every job still working (GET /v1/train/jobs, every 3 s on that tab: step,
// percent, time left, a link to where it lives): the training queue's jobs, and beside it the
// Dataset tab's exports and X-AnyLabeling imports, a benchmark run and model exports; the count
// on that tab's label, and the URL. A form's answer is the tab itself, rendered at the form's POST address: the address
// bar goes to the tab's own URL, so a reload shows it again instead of asking to resubmit. An old
// /training#import link (the redirect keeps its fragment) goes on to the card's new tab, and a
// #fragment inside a closed section opens it.
//
// TrainArea (first part, nothing done at load, so node loads it; tests/it/js_training.rs): what the
// tab scripts share. The Import, Auto-label, Train and Models tabs each have their own script
// (train-import.js, train-autolabel.js, train-train.js, train-models.js), loaded at the end of the
// page; they call:
// - asJson(res): an answer's JSON body (an error answer keeps the server's own words); a body
//   that is not JSON (a proxy's HTML 502) rejects with "HTTP <status>" instead of the parser's
//   message. checked(form, name): the value of a checked radio. toggle(form, name, attr): shows
//   the `[attr]` blocks whose list names the checked radio's value (hidden ones disabled).
// - restoreForm(id): the Import, Auto-label and Train forms keep what was typed across the page
//   the server answers with (sessionStorage, once); returns the restored picked classes or null.
//   openInvalid(): an invalid field inside a closed <details> opens it, so its message shows.
// - alEntries(cat), alSelection(picker, catalog, CP): the auto-label class picker's entries and
//   what it holds (Import and Auto-label tabs). keepAlOptions(form) / alOptions(): the settings
//   of the last auto-label run (localStorage, else the form's defaults), for the run an import
//   chains.
// - jobs(query): GET /v1/train/jobs (with `query`, e.g. "tasks=busy"): resolves {active, queued,
//   all (active then queued), task, tasks}; rejects on an answer that is not ok, not JSON or holds
//   no job list, so a PrismPoll chain built on it backs off. The Jobs tab's list, the Dataset
//   tab's export lines and the Refine tab's check follow it.
// - follow(): the Jobs card of the four tabs (the Models list shares parts): Remove with files
//   asks first, each run's Log box keeps its open state, the run metrics (train-metrics.js) and
//   settings lines, the early-stop notice's Continue, and live progress of the queued and running
//   runs and import / auto-label jobs: GET /v1/training?poll=1 every 3 s while one is busy and the
//   tab shows (a PrismPoll chain), then the page goes to its own address once one finishes, so the
//   results show.
(function (root) {
  "use strict";

  function asJson(res) {
    return res.json().catch(function () { throw new Error("HTTP " + res.status); });
  }

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

  // The form `id` keeps what was typed across the page the server answers with (it renders the
  // defaults, also after "Not queued: ..."): a submission that goes ahead stashes the visible
  // fields in sessionStorage (`formdata`, after the submit handlers added the picked classes),
  // the next load puts them back once, before the tab script's toggles read them. Only fields
  // that existed at submit are touched (a folder saved meanwhile keeps its default); hidden
  // inputs are not restored. Returns the picked classes the submission posted, or null.
  function restoreForm(id) {
    var form = document.getElementById(id);
    if (!form) return null;
    var KEY = "bop.training.form." + id;
    var keyOf = function (el, seen) {
      if (el.type === "checkbox" || el.type === "radio") return el.name + "\u0000" + el.value;
      var n = seen[el.name] = (seen[el.name] || 0) + 1;
      return el.name + "\u0000#" + n;
    };
    var kept = function (el) {
      return el.name && el.type !== "hidden" && el.type !== "file" && el.type !== "submit" && el.type !== "button";
    };
    form.addEventListener("formdata", function () {
      var v = {}, seen = {}, classes = [];
      Array.prototype.forEach.call(form.elements, function (el) {
        if (el.type === "hidden" && el.name === "classes") classes.push(el.value);
        if (!kept(el)) return;
        var k = keyOf(el, seen);
        if (el.type === "checkbox" || el.type === "radio") v[k] = el.checked;
        else if (el.multiple) v[k] = Array.prototype.filter.call(el.options, function (o) { return o.selected; }).map(function (o) { return o.value; });
        else v[k] = el.value;
      });
      try { window.sessionStorage.setItem(KEY, JSON.stringify({ fields: v, classes: classes })); } catch (e) { /* storage blocked */ }
    });
    var saved = null;
    try {
      saved = JSON.parse(window.sessionStorage.getItem(KEY) || "null");
      window.sessionStorage.removeItem(KEY);
    } catch (e) { saved = null; }
    if (!saved || typeof saved.fields !== "object") return null;
    var seen = {};
    Array.prototype.forEach.call(form.elements, function (el) {
      if (!kept(el)) return;
      var k = keyOf(el, seen);
      if (!Object.prototype.hasOwnProperty.call(saved.fields, k)) return;
      var x = saved.fields[k];
      if (el.type === "checkbox" || el.type === "radio") el.checked = !!x;
      else if (el.multiple && Array.isArray(x)) Array.prototype.forEach.call(el.options, function (o) { o.selected = x.indexOf(o.value) >= 0; });
      else if (typeof x === "string") el.value = x;
    });
    return Array.isArray(saved.classes) && saved.classes.length ? saved.classes : null;
  }

  // An invalid field inside a closed <details> (Sampling options, Advanced) cannot show its
  // message: open it first.
  function openInvalid() {
    document.querySelectorAll("form").forEach(function (f) {
      f.addEventListener("invalid", function (ev) {
        var d = ev.target.closest && ev.target.closest("details");
        if (d && !d.open) d.open = true;
      }, true);
    });
  }

  // The class picker's entries of catalog `cat`.
  function alEntries(cat) {
    var out = root.ClassPicker.util.mergeEntries(cat.entries || [], (cat.classes || []));
    out.forEach(function (e) { if ((cat.classes || []).indexOf(e.name) >= 0) e.in_list = true; e.hidden = (cat.hidden || []).indexOf(e.name) >= 0; });
    return out;
  }
  // What the picker `picker` holds, for the Preview and the auto-label an import chains.
  function alSelection(picker, catalog, CP) {
    var names = picker.getSelected();
    return {
      names: names,
      missing: CP.util.missingFrom(names, catalog.classes || []).length > 0,
      listed: function () { return names.filter(function (n) { return (catalog.classes || []).indexOf(n) >= 0; }); },
      ensure: function () { return CP.ensureInList(names, catalog); }
    };
  }

  // The settings of an auto-label run, kept for the run an import chains on the Import tab: the
  // form's fields without the frames, classes and confirmations, which are that form's own.
  var AL_OPTIONS = "bop.training.alOptions";
  var AL_DEFAULTS = [
    ["mode", "labelers"], ["gdino", "tiny"], ["vlm", "2b"], ["vlm_batch", "on"], ["vlm_min_on", "on"],
    ["vlm_min_score", "0.15"], ["rfdetr", "on"], ["siglip", "on"], ["prefetch", "on"],
    ["run_options", "1"], ["keep_on_cancel", "on"]
  ];
  var AL_NOT_OPTIONS = ["classes", "confirm_large", "status", "scene", "camera", "label", "auto", "limit", "frames", "again", "missing_only"];
  function keepAlOptions(form) {
    form.addEventListener("formdata", function (ev) {
      var kept = [];
      ev.formData.forEach(function (value, name) {
        if (typeof value === "string" && AL_NOT_OPTIONS.indexOf(name) < 0) kept.push([name, value]);
      });
      try { window.localStorage.setItem(AL_OPTIONS, JSON.stringify(kept)); } catch (e) { /* defaults next time */ }
    });
  }
  function alOptions() {
    var kept = null;
    try { kept = JSON.parse(window.localStorage.getItem(AL_OPTIONS) || "null"); } catch (e) { kept = null; }
    return Array.isArray(kept) && kept.length ? kept : AL_DEFAULTS;
  }

  function jobs(query) {
    return fetch("/v1/train/jobs" + (query ? "?" + query : ""), { headers: { Accept: "application/json" }, cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (j) {
        if (!j || !j.jobs || !Array.isArray(j.jobs.active) || !Array.isArray(j.jobs.queued)) throw new Error("no job list in the answer");
        return { active: j.jobs.active, queued: j.jobs.queued, all: j.jobs.active.concat(j.jobs.queued), task: j.task, tasks: j.tasks };
      });
  }

  function follow() {
    // Deleting a run's files asks first, naming the size (the form's data-confirm).
    document.querySelectorAll("form[data-confirm]").forEach(function (f) {
      f.addEventListener("submit", function (e) {
        if (!window.confirm(f.dataset.confirm)) e.preventDefault();
      });
    });

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
      var known = (data.metrics || metricsState.metrics || []).map(function (m) { return m.id; }).filter(function (id) { return seen[id]; });
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
        // Same history and columns: only the tiles move (batch, learning rate, time left).
        if (!force && host.dataset.sig === sig) {
          if (r && r.progress && window.TrainMetrics.refresh) {
            window.TrainMetrics.refresh(host, r.progress, metricsState.metrics, { running: r.state === "running" });
          }
          return;
        }
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
    // The run's settings under its progress text, joined with " · ": device, batch (and the OOM
    // retry that lowered it), input size, learning rate and schedule, epoch cap, patience, the
    // metric selected by (when not mAP), seed, scene, data loader workers. From progress.settings
    // (what the job resolved) and the trainer's
    // start event (batch_size, imgsz); a run queued before those existed falls back to its
    // request's recipe. Missing parts are left out; nothing known = no line.
    function lrText(v) { return v === 0 ? "0" : (+v.toPrecision(4)).toExponential(); }
    var isNum = function (v) { return typeof v === "number" && !isNaN(v); };
    function metricShort(id) {
      var m = (metricsState.metrics || []).filter(function (x) { return x.id === id; })[0];
      return (m && m.short) || id;
    }
    function settingsParts(r) {
      var p = r.progress || {}, s = p.settings || null, rec = (r.request && r.request.recipe) || {};
      var out = [];
      var dev = r.device || p.device || (s && s.gpu) || "";
      if (dev) out.push(dev);
      var want = s && isNum(s.batch) ? s.batch : rec.batch;
      if (isNum(p.batch_size)) out.push("batch " + p.batch_size + (isNum(want) && p.batch_size < want ? " (OOM retry from " + want + ")" : ""));
      else if (isNum(want)) out.push("batch " + want);
      var px = isNum(p.imgsz) ? p.imgsz : rec.imgsz;
      if (isNum(px)) out.push(px + " px");
      if (s) {
        var lr = isNum(s.lr) ? "lr " + lrText(s.lr) : "lr auto";
        if (s.lr_schedule) lr += " " + s.lr_schedule + (isNum(s.lr_min_factor) ? " → " + (+s.lr_min_factor.toPrecision(3)) + "×" : "");
        if (isNum(s.lr_encoder)) lr += ", enc " + lrText(s.lr_encoder);
        out.push(lr);
      } else if (isNum(rec.lr)) {
        out.push("lr " + lrText(rec.lr));
      }
      var cap = s && isNum(s.epochs) ? s.epochs : isNum(rec.epochs) ? rec.epochs : p.epochs;
      if (isNum(cap) && cap > 0) out.push("cap " + cap + " ep");
      var pat = s && isNum(s.patience) ? s.patience : p.patience;
      if (isNum(pat)) out.push(pat > 0 ? "patience " + pat + (s && s.patience_doubled ? " (×2, small val)" : "") : "early stop off");
      var select = s ? s.select_metric : rec.select_metric;
      // "selected by F2 over car, person": the classes with enough validation boxes (the support
      // event, TrainMetrics.selectedBy); an older record names the metric only.
      var TM = window.TrainMetrics;
      var by = !p.select_fallback && TM && TM.selectedBy ? TM.selectedBy(p, function (id) { return { short: metricShort(id) }; }) : "";
      if (p.select_fallback) out.push("selected by mAP (" + metricShort(p.select_fallback.metric) + " failed at epoch " + p.select_fallback.epoch + ")");
      else if (by) out.push("selected " + by);
      else if (select && select !== "map") out.push("selected by " + metricShort(select));
      var seed = s && isNum(s.seed) ? s.seed : r.request && isNum(r.request.train_seed) ? r.request.train_seed : null;
      if (seed !== null) out.push("seed " + seed);
      if (s && typeof s.scene === "string" && s.scene && s.scene !== "all") out.push(s.scene);
      // training.ignore: said only when it is not the default (records before it are drop-frame).
      if (s && s.ignore === "drop-frame") out.push("frames with a too-small object left out");
      if (s && isNum(s.workers)) out.push("workers " + s.workers);
      return out;
    }
    function settingsLine(tr, r) {
      var desc = tr.querySelector('[data-field="describe"]');
      if (!desc) return;
      var line = tr.querySelector(".tr-settings");
      var text = settingsParts(r).join(" · ");
      if (!line) {
        if (!text) return;
        line = document.createElement("p");
        line.className = "muted small tr-settings";
        desc.parentNode.insertBefore(line, desc.nextSibling);
      }
      line.textContent = text;
      line.hidden = !text;
    }
    function renderSettings(data) {
      var t = document.getElementById("training-runs");
      if (!t || !data) return;
      (data.runs || []).forEach(function (r) {
        var tr = t.querySelector('tr[data-run="' + r.id + '"]');
        if (tr) settingsLine(tr, r);
      });
    }
    // The light answer of GET /v1/training?poll=1 (src/server/training/poll.rs): the epochs this
    // page holds are named in `have` and not sent again; TrainMetrics.poll puts the whole history
    // back on each run, so the answer reads as a full one. No capability (the Auto-label tab's GPU
    // fit check fetches it once), no losses, logs only of busy jobs, no metrics catalog after the
    // first (renderMetrics keeps it). Without train-metrics.js (no run on the page) nothing is
    // held and every history comes whole.
    var pollStore = {};
    function fetchTraining() {
      var P = window.TrainMetrics && window.TrainMetrics.poll;
      var have = P ? P.have(pollStore) : "";
      // Any `have` (an empty one too) also says the page holds the metrics catalog: not sent again.
      var holds = have || (metricsState.metrics && metricsState.metrics.length);
      return fetch("/v1/training?poll=1" + (holds ? "&have=" + have : ""), { headers: { Accept: "application/json" }, cache: "no-store" })
        .then(function (res) {
          if (!res.ok) throw new Error("HTTP " + res.status);
          return res.json();
        })
        .then(function (data) {
          if (P) P.merge(pollStore, data.runs);
          return data;
        });
    }
    function loadMetrics() {
      if (!document.querySelector(".tr-metrics[data-run]")) return;
      fetchTraining()
        .then(function (data) { renderSettings(data); renderMetrics(data, false); })
        .catch(function () {});
    }

    // Continue hides the notice of that early-stop point (the run goes on by itself; a later
    // point shows again).
    function extendKey(el) { return "bop-extend:" + el.dataset.run + ":" + el.dataset.ext; }
    function extendDismissed(el) {
      try { return sessionStorage.getItem(extendKey(el)) === "1"; } catch (e) { return false; }
    }
    document.querySelectorAll('[data-field="extend"]').forEach(function (el) {
      if (extendDismissed(el)) el.hidden = true;
      var btn = el.querySelector('[data-action="extend-continue"]');
      if (btn) btn.addEventListener("click", function () {
        try { sessionStorage.setItem(extendKey(el), "1"); } catch (e) { /* private mode: hide only */ }
        el.hidden = true;
      });
    });

    if (Object.keys(busy).length === 0) { loadMetrics(); return; }
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
        // The actual batch / input size arrive with the trainer's start event: kept current.
        if (attr === "run") settingsLine(tr, r);
        // The dataset's warnings arrive once the dataset is assembled.
        var warnList = attr === "run" ? tr.querySelector('[data-field="warnings"]') : null;
        if (warnList) {
          var ws = (r.dataset && r.dataset.warnings) || [];
          if (warnList.childElementCount !== ws.length) {
            warnList.replaceChildren();
            ws.forEach(function (w) {
              var li = document.createElement("li");
              li.textContent = w;
              warnList.appendChild(li);
            });
          }
          warnList.hidden = !ws.length;
        }
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
        // Finish now: a run in its training stage that has not stopped or started exporting.
        var finish = tr.querySelector('[data-field="finish"]');
        if (finish) {
          var p = r.progress || {};
          finish.hidden = !((r.state === "running" || r.state === "paused") && r.stage === "train" && !p.stopped && !p.phase && !r.retrying);
        }
        // An early-stop point passed while validation still improves: Stop now / Continue.
        var ext = tr.querySelector('[data-field="extend"]');
        if (ext) {
          var n = (r.progress || {}).extension;
          var show = !!n && finish && !finish.hidden;
          if (show) {
            ext.dataset.ext = String(n.extension);
            set("extend-text", n.text);
            set("extend-stop", "Stop now and keep epoch " + n.keep_epoch);
          }
          ext.hidden = !show || extendDismissed(ext);
        }
        if (busy[r.id] && !isBusy(r.state)) finished = true;
      });
      return finished;
    }
    // One tick: false once a job this page showed busy has finished and the page moves on (the
    // loop stops). A chained loop, never an interval: the next request leaves 3 s after this one
    // answered, none while the tab is hidden, and one at once when it shows again, so a run that
    // finished meanwhile still navigates.
    function tick() {
      return fetchTraining().then(function (data) {
        var a = update(runs, "run", data.runs);
        var b = update(tasks, "task", data.tasks);
        renderMetrics(data, false);
        // States and progress changed in place: the user's sort (sort.js) again.
        if (window.PrismSort) [runs, tasks].forEach(function (t) { if (t) window.PrismSort.apply(t); });
        if (!(a || b)) return true;
        // Not reload(): the page may be the answer to a POST, and reloading it would ask to
        // resubmit (and queue the job again). The poll navigates, not the user: no activity.
        PrismPoll.navigate(window.location.pathname + window.location.hash);
        return false;
      });
    }
    PrismPoll.create({ run: tick, period: 3000, more: function (go) { return go !== false; } }).start();
  }

  root.TrainArea = {
    asJson: asJson, checked: checked, toggle: toggle, restoreForm: restoreForm, openInvalid: openInvalid,
    alEntries: alEntries, alSelection: alSelection, keepAlOptions: keepAlOptions, alOptions: alOptions,
    jobs: jobs, follow: follow
  };
})(typeof globalThis !== "undefined" ? globalThis : window);

(function () {
  "use strict";
  if (typeof document === "undefined") return; // loaded by a node test: only TrainArea
  var strip = document.getElementById("train-strip");
  if (!strip) return;
  var tab = strip.dataset.tab;
  var home = "/train/" + tab;

  // Where the cards of the old Training page went.
  var MOVED = {
    import: "/train/import", autolabel: "/train/autolabel", review: "/train/review",
    use: "/train/models#use", compare: "/train/models#compare"
  };
  if (tab === "train" && MOVED[location.hash.slice(1)]) {
    location.replace(MOVED[location.hash.slice(1)]);
    return;
  }
  if (location.pathname !== home) {
    try { history.replaceState(null, "", home + location.hash); } catch (e) { /* stays */ }
  }

  function openTarget() {
    var id = location.hash.slice(1);
    var el = id && document.getElementById(id);
    if (!el) return;
    var opened = false;
    for (var d = el.closest("details"); d; d = d.parentElement && d.parentElement.closest("details")) {
      if (!d.open) { d.open = true; opened = true; }
    }
    if (opened) el.scrollIntoView();
  }
  document.addEventListener("DOMContentLoaded", openTarget);
  window.addEventListener("hashchange", openTarget);

  // On a phone the tabs scroll sideways: the current one in view.
  var nav = document.querySelector("nav.train-tabs"), active = nav && nav.querySelector("a.active");
  if (active) nav.scrollLeft = active.offsetLeft - nav.offsetLeft - (nav.clientWidth - active.offsetWidth) / 2;

  var list = document.getElementById("train-counts");
  var nextText = document.getElementById("train-next-text");
  var nextLink = document.getElementById("train-next-link");
  var nextLine = nextText.closest(".train-next");
  // This script runs above the tab's content: the Jobs card's list is looked up once it exists.
  var jobsBox = null, jobsEmpty = null;
  var jobsCount = document.getElementById("train-jobs-count");

  function item(tabId, number, label, note) {
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = "/train/" + tabId;
    var b = document.createElement("strong");
    b.textContent = number;
    a.appendChild(b);
    a.appendChild(document.createTextNode(" " + label));
    if (note) {
      var s = document.createElement("span");
      s.className = "muted";
      s.textContent = " " + note;
      a.appendChild(s);
    }
    li.appendChild(a);
    return li;
  }

  function draw(j) {
    var s = j.summary, n = j.next;
    var unknown = s.counted ? null : "…";
    list.replaceChildren(
      item("collect", s.frames_collected, "collected"),
      item("review", unknown || s.frames_to_review, "to review", s.boxes_to_review ? "(" + s.boxes_to_review + " boxes)" : ""),
      item("dataset", unknown || s.frames_reviewed, "reviewed"),
      item("dataset", s.exports, s.exports === 1 ? "export" : "exports"),
      item("train", s.runs, s.runs === 1 ? "run" : "runs"),
      item("models", s.models, s.models === 1 ? "model" : "models")
    );
    nextText.textContent = n.text;
    nextLink.textContent = n.label;
    nextLink.href = n.link;
    // On its own tab the link is hidden, unless it asks for a list or a card the page is not on
    // (Review's `?auto=refine`, Refine's `#quick-check`).
    var at = n.link.search(/[?#]/), more = at < 0 ? "" : n.link.slice(at);
    nextLink.hidden = n.tab === tab && (!more || more === location.search + location.hash);
    // The Jobs tab shows the job itself.
    nextLine.hidden = n.key === "job" && tab === "jobs";
    strip.hidden = false;
  }

  // The rows stay in place between polls, so a hover or keyboard focus survives.
  var rows = {};
  function span(cls, parent) {
    var el = document.createElement("span");
    el.className = cls;
    parent.appendChild(el);
    return el;
  }
  function jobRow() {
    var a = document.createElement("a");
    a.className = "tj";
    var head = span("tj-head", a);
    var f = { a: a, kind: span("tj-kind", head), title: span("tj-title", head), state: span("tj-state", a), open: span("tj-open", a) };
    var meter = span("tj-meter", a);
    f.bar = document.createElement("progress");
    f.bar.max = 100;
    meter.appendChild(f.bar);
    f.pct = span("tj-pct", meter);
    f.time = span("tj-time", a);
    f.step = span("tj-step", a);
    return f;
  }
  function fillRow(f, j) {
    var state = j.queued ? "Queued" : j.paused ? "Paused" : "Running";
    f.a.href = j.link;
    f.a.dataset.state = state.toLowerCase();
    f.a.setAttribute("aria-label", j.kind + " " + j.title + ", " + state + ", " + (j.queued ? "" : j.percent + "%, ") + j.time + ". Open the " + j.tab + " tab");
    f.kind.textContent = j.kind;
    f.title.textContent = j.title;
    f.title.title = j.title;
    f.state.textContent = state;
    f.time.textContent = j.time;
    f.bar.value = j.percent;
    f.pct.textContent = j.percent + "%";
    var where = j.step && j.phase ? j.step + ": " + j.phase : j.step || j.phase;
    var text = [where, j.detail].filter(Boolean).join(" · ");
    f.step.textContent = text;
    f.step.hidden = !text;
    f.open.textContent = j.tab + " ›";
  }
  function drawJobs(all) {
    var seen = {};
    all.forEach(function (j) {
      seen[j.id] = true;
      if (!rows[j.id]) rows[j.id] = jobRow();
      fillRow(rows[j.id], j);
    });
    Object.keys(rows).forEach(function (id) {
      if (!seen[id]) { rows[id].a.remove(); delete rows[id]; }
    });
    all.forEach(function (j) { jobsBox.appendChild(rows[j.id].a); });
    jobsBox.hidden = !all.length;
    jobsEmpty.hidden = !!all.length;
  }

  // One chain per list (PrismPoll: a chained timeout, never two requests of a list in flight, none
  // while the tab is hidden). A refresh asks the same chain to run now; it never starts another.
  var onQueue = null, jobsNote = null;
  // One line above the list while the server does not answer (the list keeps its last state).
  function jobsOutage(text) {
    if (!jobsBox) return;
    if (!jobsNote) {
      if (!text) return;
      jobsNote = document.createElement("p");
      jobsNote.className = "muted";
      jobsNote.setAttribute("role", "status");
      jobsBox.parentNode.insertBefore(jobsNote, jobsBox);
    }
    jobsNote.textContent = text;
    jobsNote.hidden = !text;
  }
  // Rejects on an answer that is not the list (HTTP error, HTML from a proxy): PrismPoll then backs
  // off (6, 12, ... 30 s) instead of asking every 3 s through an outage, and onError says so.
  function loadJobs() {
    return TrainArea.jobs()
      .then(function (j) {
        jobsOutage("");
        var all = j.all;
        var key = all.map(function (x) { return x.id; }).join();
        jobsCount.textContent = all.length;
        jobsCount.hidden = !all.length;
        if (jobsBox) drawJobs(all);
        // A job ended or started: the counts and the next step moved too.
        if (onQueue !== null && key !== onQueue) summaryPoller.kick();
        onQueue = key;
        return !!jobsBox;
      });
  }
  // Only the Jobs tab keeps following the list; the other tabs read it when asked (the count on its label).
  var jobsPoller = PrismPoll.create({
    run: loadJobs, period: 3000, more: function (again) { return again; },
    onError: function () {
      if (!jobsBox) { jobsPoller.stop(); return; }
      jobsOutage("The server is not answering; this list may be out of date. Trying again shortly.");
    }
  });

  function load() {
    return fetch("/v1/train/summary", { headers: { Accept: "application/json" }, cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.summary) return false;
        draw(j);
        // A first count of a large folder is still running.
        return !j.summary.counted;
      })
      .catch(function () { /* the strip stays as it was */ return false; });
  }
  var summaryPoller = PrismPoll.create({ run: load, period: 2000, more: function (again) { return again; } });
  function loadAll() { summaryPoller.kick(); jobsPoller.kick(); }
  document.addEventListener("DOMContentLoaded", function () {
    jobsBox = document.getElementById("train-jobs");
    jobsEmpty = document.getElementById("train-jobs-empty");
    if (jobsBox) jobsPoller.kick();
  });
  loadAll();
  // Pages that change the counts say so.
  window.addEventListener("train-area:refresh", loadAll);
  // A chain that is running resumes itself when the tab shows; a parked one is asked here.
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) return;
    [summaryPoller, jobsPoller].forEach(function (p) { if (p.state().parked) p.kick(); });
  });
  setInterval(function () { if (!document.hidden) loadAll(); }, 30000);
})();
