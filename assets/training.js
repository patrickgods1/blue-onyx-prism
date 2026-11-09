// The Import, Auto-label, Train and Models tabs of the Train area. One file for the four: each block
// looks up its form or card and returns when this tab does not have it. The forms' show / hide
// (import source, auto-label mode, night augmentation, the model name with several models ticked;
// without JS every field shows and works), live progress of the queued and running training runs
// and import / auto-label jobs (polls GET /v1/training?poll=1 every 3 s while one is busy and the
// tab shows, chained) and a reload when one finishes, so the results show.
// Auto-label after import: a ticked import posts the settings of the last Auto-label run (kept in
// localStorage, else the form's defaults) as `al.<name>` hidden inputs, with the classes picked on
// the Import tab; the box and its follow-up options are remembered in localStorage. A link to
// /train/autolabel?<filters> (Review's "Auto-label these frames", a Recent imports row) fills the
// form's Which frames; /train/train?dataset=<folder> ticks that export.
// Train: the runs line (one run per ticked scene per model, named before queuing), the class
// picker of the ticked folders (counts per scene, warnings), the balance preview (what the
// Balance fieldset does to each run's training frames) and Finish now while training, plus the
// Stop now / Continue notice of an early-stop point passed while the training loss still falls.
// Each run row gets a settings line (device, batch, input size, learning rate and schedule,
// patience, seed, ...) and the Learning rate field's placeholder follows the ticked family.
// SAM 3: the three-step access panel of the Auto-label card (GET /v1/autolabel/sam3); its option
// in the labeler select stays disabled until the checkpoint is installed. Its threshold field
// shows only while SAM 3 is the chosen labeler.
// Auto-label frames: the line above the button says what a run looks at, and the "Which frames"
// select under Advanced writes Status and Auto (the fields the form posts;
// the select itself has no name, so the import chain's `al.*` copy and the form's restore are the
// fields as before); Custom shows them. Preview frames posts the form to
// POST /training/autolabel/preview on a click (it reads every sidecar) and draws the answer.
// Use card: the default switch's preview (what changes, copied settings, suggested threshold)
// and the confirm that names a verifier about to be replaced.
(function () {
  "use strict";

  // Set by the auto-label classes block: the picker's selection, whether some are outside the
  // class list, and the confirm that appends them (null without the picker).
  var alClasses = null;
  // Set by the train classes block: the picked classes in head order (null without the picker).
  var trainClasses = null;

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
  // The Import, Auto-label and Train forms keep what was typed across the page the server
  // answers with (it renders the defaults, also after "Not queued: ..."): a submission that goes
  // ahead stashes the visible fields in sessionStorage (`formdata`, after the submit handlers
  // added the picked classes), the next load puts them back once, before the toggles below read
  // them. Only fields that existed at submit are touched (a folder saved meanwhile keeps its
  // default); hidden inputs are not restored.
  var restoredClasses = {};
  ["import-form", "autolabel-form", "train"].forEach(function (id) {
    var form = document.getElementById(id);
    if (!form) return;
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
    if (!saved || typeof saved.fields !== "object") return;
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
    if (Array.isArray(saved.classes) && saved.classes.length) restoredClasses[id] = saved.classes;
  });

  // A link's filters into the Auto-label form, before the blocks below read its fields. Keys with a
  // field of their own fill it; the others, and a value its select does not offer, go to the filter
  // text. The note above the form says what was filled and Clear goes back to the defaults.
  (function () {
    var form = document.getElementById("autolabel-form");
    if (!form) return;
    var q = new URLSearchParams(window.location.search);
    var parts = [], said = [];
    ["status", "scene", "camera", "label", "auto", "limit"].forEach(function (k) {
      var v = q.get(k), e = form.elements[k];
      if (!v) return;
      if (e) {
        var was = e.value;
        e.value = v;
        if (e.options && e.value !== v) { e.value = was; parts.push(k + "=" + v); }
      } else parts.push(k + "=" + v);
      if (k !== "limit") said.push(k + " " + v);
    });
    ["bucket", "source", "import"].forEach(function (k) {
      var v = q.get(k);
      if (!v) return;
      parts.push(k + "=" + v);
      if (k !== "source") said.push(k + " " + v);
    });
    if (form.elements.frames && parts.length) form.elements.frames.value = parts.join(",");
    if (q.get("again") === "1" && form.elements.again) form.elements.again.checked = true;
    var note = document.getElementById("al-prefill"), text = document.getElementById("al-prefill-text");
    if (note && text && said.length) {
      text.textContent = "Filled in from your link: " + said.join(", ") + ".";
      note.hidden = false;
    }
  })();

  // A link's export: its box ticked, or the Another folder field when the list does not offer it.
  (function () {
    var form = document.getElementById("train");
    var want = new URLSearchParams(window.location.search).get("dataset");
    if (!form || !want) return;
    var boxes = form.querySelectorAll('input[name="dataset"]');
    var hit = Array.prototype.filter.call(boxes, function (b) { return b.value === want && !b.disabled; })[0];
    if (hit) hit.checked = true;
    else if (form.elements.dataset_path) form.elements.dataset_path.value = want;
  })();

  // "Select best by": the hint of the chosen metric (the server's list, on each option).
  (function () {
    var sel = document.getElementById("train-select-metric");
    var hint = document.getElementById("train-select-hint");
    if (!sel || !hint) return;
    var show = function () {
      var o = sel.options[sel.selectedIndex];
      if (o) hint.textContent = o.getAttribute("data-hint") || "";
    };
    sel.addEventListener("change", show);
    show();
  })();

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
  // Auto-label: Qwen3-VL sizes the labelers' GPU cannot hold are disabled with the reason. Each
  // option carries what it needs (data-need-mb, the server's rule in training/autolabel/memory.rs,
  // which refuses such a request); the GPU's memory comes from the capability of /v1/training.
  (function () {
    var form = document.getElementById("autolabel-form");
    var note = document.getElementById("al-fit");
    if (!form || !note || !form.elements.vlm) return;
    var vlm = form.elements.vlm, gpu = form.elements.gpu, cap = null;
    var chosen = function () {
      if (!cap) return null;
      var a = cap.accelerator, want = gpu ? gpu.value : "";
      if (want) (cap.cuda_options || []).forEach(function (o) { if ("cuda:" + o.ordinal === want) a = o; });
      return a;
    };
    var gb = function (mb) { return (mb / 1024).toFixed(1) + " GB"; };
    var update = function () {
      var a = chosen(), mem = a ? (a.vram_mb || a.memory_mb || 0) : 0, big = [];
      Array.prototype.forEach.call(vlm.options, function (o) {
        var need = parseInt(o.getAttribute("data-need-mb") || "0", 10);
        if (o.dataset.label === undefined) o.dataset.label = o.textContent;
        var over = mem > 0 && need > mem;
        o.disabled = over;
        o.textContent = over ? o.dataset.label + " \u2013 needs " + gb(need) : o.dataset.label;
        if (over) big.push(o.dataset.label.replace(/ \(.*$/, ""));
      });
      if (vlm.selectedOptions.length && vlm.selectedOptions[0].disabled) {
        var fit = Array.prototype.filter.call(vlm.options, function (o) { return !o.disabled; });
        vlm.value = fit.length ? fit[0].value : "off";
        vlm.dispatchEvent(new Event("change", { bubbles: true }));
      }
      note.hidden = !big.length;
      note.textContent = big.length
        ? big.join(" and ") + " need more GPU memory than " + a.name + " has (" + gb(mem) + ")."
        : "";
    };
    if (gpu) gpu.addEventListener("change", update);
    fetch("/v1/training", { headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) { cap = j && j.capability; update(); })
      .catch(function () {});
  })();
  // Auto-label: the SAM 3 threshold only while SAM 3 is the chosen labeler. Hidden, it is disabled:
  // not posted, not validated, not copied into an import's `al.*` fields. Like the block above this
  // runs after the mode toggle, which re-enables every field of the shown mode.
  (function () {
    var form = document.getElementById("autolabel-form");
    var field = document.getElementById("al-sam3-threshold-field");
    if (!form || !field || !form.elements.gdino || !form.elements.sam3_threshold) return;
    var seat = form.elements.gdino, input = form.elements.sam3_threshold;
    var update = function () {
      var on = checked(form, "mode") === "labelers" && seat.value === "sam3";
      field.hidden = !on;
      input.disabled = !on;
    };
    seat.addEventListener("change", update);
    form.querySelectorAll('input[name="mode"]').forEach(function (r) { r.addEventListener("change", update); });
    update();
  })();

  // SAM 3 is gated: request access, say so (POST .../download, which also adopts a folder already
  // on disk by hash), then sign in with the command shown. The server decides the state; this
  // only draws it, and polls while something runs.
  (function () {
    var panel = document.getElementById("al-sam3");
    if (!panel) return;
    var API = "/v1/autolabel/sam3";
    var $ = function (id) { return document.getElementById("al-sam3-" + id); };
    var option = document.getElementById("al-gdino-sam3");
    var select = document.getElementById("al-gdino");
    var download = $("download"), retry = $("retry"), remove = $("remove"), progress = $("progress");
    var msg = $("msg"), command = $("command"), copied = $("copied");
    var SUMMARY = {
      not_requested: "needs access", needs_login: "needs a login", token_unreadable: "login unreadable",
      needs_access: "waiting for access", checking: "checking the files", downloading: "downloading",
      installed: "installed", failed: "failed"
    };
    var LOGIN = ["needs_login", "needs_access", "token_unreadable", "failed"];
    var timer = null, sizeText = "";
    var draw = function (s) {
      panel.hidden = false;
      var busy = s.state === "checking" || s.state === "downloading";
      var installed = s.state === "installed";
      sizeText = s.size_text || sizeText;
      $("summary").textContent = "(" + (SUMMARY[s.state] || s.state) + ")";
      $("link").href = s.repo_url;
      $("license").textContent = s.license;
      if (option) {
        option.disabled = !installed;
        option.textContent = installed ? "SAM 3" : "SAM 3 (gated, needs the download below)";
        if (!installed && select && select.value === "sam3") {
          select.value = select.options[0].value;
          select.dispatchEvent(new Event("change", { bubbles: true }));
        }
      }
      download.textContent = s.parts_on_disk ? "Check the files on disk" : "I have access, download (" + s.size_text + ")";
      download.hidden = installed;
      download.disabled = busy;
      progress.hidden = !busy;
      if (typeof s.progress === "number") progress.value = s.progress; else progress.removeAttribute("value");
      $("login").hidden = LOGIN.indexOf(s.state) < 0;
      command.textContent = s.command;
      $("note").textContent = s.command_note;
      msg.textContent = s.message || "";
      msg.className = "form-msg" + (LOGIN.indexOf(s.state) >= 0 ? " error-text" : "");
      $("source").hidden = installed;
      $("source").textContent = s.token_source ? "Token from " + s.token_source + "." : "No token set up.";
      $("override").hidden = !s.token_env_overrides;
      $("apple").hidden = !s.apple;
      remove.hidden = !(installed || s.parts_on_disk);
      remove.disabled = busy || s.in_use;
      remove.title = s.in_use ? "An auto-label run is using it" : "";
      if (timer) { clearTimeout(timer); timer = null; }
      // A poll while it is busy: it does not keep the session alive (live.js).
      if (busy) timer = setTimeout(function () {
        if (window.PrismPoll && window.PrismPoll.background) window.PrismPoll.background(load); else load();
      }, 2000);
    };
    var fail = function (text) {
      msg.textContent = text;
      msg.className = "form-msg error-text";
    };
    function load() {
      fetch(API, { headers: { Accept: "application/json" } })
        .then(function (res) { return res.json(); })
        .then(function (s) { if (s.state) draw(s); })
        .catch(function () {});
    }
    // The answer of a refused action is a state too (409 needs_login, ...), so it is drawn the same.
    var act = function (path, button) {
      button.disabled = true;
      fetch(API + path, { method: "POST", headers: { Accept: "application/json" } })
        .then(function (res) { return res.json(); })
        .then(function (s) {
          button.disabled = false;
          if (s.state) draw(s); else fail(s.message || "The server gave no answer.");
        })
        .catch(function (e) { button.disabled = false; fail(e.message); });
    };
    download.addEventListener("click", function () { act("/download", download); });
    retry.addEventListener("click", function () { act("/download", retry); });
    remove.addEventListener("click", function () {
      if (window.confirm("Remove SAM 3 (" + sizeText + ") from this machine?")) act("/remove", remove);
    });
    $("copy").addEventListener("click", function () {
      var said = function (t) { copied.textContent = t; };
      var byHand = function () {
        // No clipboard API outside a secure context (or when the browser refuses it).
        var range = document.createRange();
        range.selectNodeContents(command);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        try { said(document.execCommand("copy") ? "copied" : "selected: press Ctrl+C"); } catch (e) { said("selected: press Ctrl+C"); }
      };
      if (navigator.clipboard) navigator.clipboard.writeText(command.textContent).then(function () { said("copied"); }, byHand);
      else byHand();
    });
    load();
  })();

  // The class picker's entries of catalog `cat`.
  function alEntries(cat) {
    var out = window.ClassPicker.util.mergeEntries(cat.entries || [], (cat.classes || []));
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

  // Auto-label classes on the Import tab: the same picker, for a run the import chains.
  (function () {
    var holder = document.getElementById("import-al-picker"), CP = window.ClassPicker;
    if (!holder || !CP) return;
    var picker = null, catalog = null;
    CP.fetchCatalog().then(function (cat) {
      catalog = cat;
      picker = CP.create(holder, {
        entries: alEntries(cat), selected: CP.util.defaultSelection(cat), mode: "compact", label: "Classes", allowAdd: true,
        quickSets: CP.util.quickSetsFor(cat)
      });
    }).catch(function (e) {
      holder.textContent = "";
      var p = document.createElement("p");
      p.className = "muted small";
      p.textContent = "The class list could not be read (" + e.message + "); the labelers use the class list and training.synonyms.";
      holder.appendChild(p);
    });
    alClasses = function () { return picker ? alSelection(picker, catalog, CP) : null; };
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
    CP.fetchCatalog().then(function (cat) {
      catalog = cat;
      picker = CP.create(holder, {
        // A phone gets the drop-down: the chips of every class would push the form a screen down.
        entries: alEntries(cat), selected: restoredClasses["autolabel-form"] || CP.util.defaultSelection(cat),
        mode: window.matchMedia("(max-width: 700px)").matches ? "compact" : "full", label: "Classes", emptyText: "your class list", allowAdd: true,
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
    if (MP) {
      var syncMissing = MP.util.missingToggle(form);
      form.querySelectorAll('input[name="mode"]').forEach(function (r) { r.addEventListener("change", syncMissing); });
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
    alClasses = function () { return picker ? alSelection(picker, catalog, CP) : null; };
    var postJson = function (url, body) {
      return fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body)
      }).then(function (res) {
        return res.json().catch(function () { return null; }).then(function (j) {
          if (!res.ok || !j || j.success === false) throw new Error((j && j.message) || "HTTP " + res.status);
          return j;
        });
      });
    };
    // "Hide the classes left unticked": listed, wanted classes the run skips stop being wanted
    // (their pending boxes are rejected by the class save), then the reviewed frames that still
    // hold accepted boxes of them ask. Cancel only skips them in this run.
    var hideUnticked = function (names) {
      var box = document.getElementById("al-hide-unticked");
      var list = catalog.classes || [], hidden = catalog.hidden || [];
      var skipped = !box || !box.checked ? [] : list.filter(function (c) { return names.indexOf(c) < 0 && hidden.indexOf(c) < 0; });
      if (!skipped.length || !window.confirm("Hide and auto-reject " + skipped.join(", ") + " in Review? (Cancel = only skip them in this run)")) return Promise.resolve();
      return postJson("/v1/review/classes", { classes: list.slice(), hidden: hidden.concat(skipped) }).then(function (j) {
        catalog.classes = j.classes || list;
        if (j.hidden) catalog.hidden = j.hidden;
        var r = j.reviewed_with_unwanted;
        if (!r || !(r.frames > 0)) return null;
        if (!window.confirm(r.frames + " reviewed frame(s) have " + r.boxes + " accepted box(es) of " + (r.classes || []).join(", ") +
          ". Reject those too? They are kept as evidence (reason other_class) and the frames stay reviewed; Cancel keeps them as training labels.")) return null;
        return postJson("/v1/review/reject-unwanted-reviewed", {});
      });
    };
    form.addEventListener("submit", function (ev) {
      if (mode() === "model" && !modelProblemOk(ev)) return;
      if (!picker) { fields([]); return; }
      var names = picker.getSelected();
      if (!names.length) { ev.preventDefault(); say("Choose at least one class to look for.", true); return; }
      say("", false);
      fields(names);
      if (confirmed) { confirmed = false; return; }
      var hide = document.getElementById("al-hide-unticked");
      if (!CP.util.missingFrom(names, catalog.classes || []).length && !(hide && hide.checked)) return;
      ev.preventDefault();
      var by = ev.submitter;
      hideUnticked(names).then(function () { return CP.ensureInList(names, catalog); }).then(function () {
        confirmed = true;
        if (form.requestSubmit) form.requestSubmit(by || undefined); else form.submit();
      }).catch(function (e) { say(e.cancelled ? "Not queued: the class list was left as it is." : e.message, true); });
    });
  })();

  // The settings of an auto-label run, kept for the run an import chains on the Import tab: the
  // form's fields without the frames, classes and confirmations, which are that form's own.
  var AL_OPTIONS = "bop.training.alOptions";
  var AL_DEFAULTS = [
    ["mode", "labelers"], ["gdino", "tiny"], ["vlm", "2b"], ["vlm_batch", "on"], ["vlm_min_on", "on"],
    ["vlm_min_score", "0.15"], ["rfdetr", "on"], ["siglip", "on"], ["prefetch", "on"],
    ["run_options", "1"], ["keep_on_cancel", "on"]
  ];
  var AL_NOT_OPTIONS = ["classes", "confirm_large", "status", "scene", "camera", "label", "auto", "limit", "frames", "again", "missing_only"];
  (function () {
    var form = document.getElementById("autolabel-form");
    if (!form) return;
    form.addEventListener("formdata", function (ev) {
      var kept = [];
      ev.formData.forEach(function (value, name) {
        if (typeof value === "string" && AL_NOT_OPTIONS.indexOf(name) < 0) kept.push([name, value]);
      });
      try { window.localStorage.setItem(AL_OPTIONS, JSON.stringify(kept)); } catch (e) { /* defaults next time */ }
    });
  })();
  function alOptions() {
    var kept = null;
    try { kept = JSON.parse(window.localStorage.getItem(AL_OPTIONS) || "null"); } catch (e) { kept = null; }
    return Array.isArray(kept) && kept.length ? kept : AL_DEFAULTS;
  }

  // "Which frames" is the one writer of Status and Auto while it names a preset, and the model
  // mode's First pass / Self-training check only moves it. On load it is worked out from the
  // fields, so a restored form shows what it holds. The fields only Custom uses are disabled under
  // a preset: a stale scene must not narrow a hidden filter. Status and Auto stay enabled, hidden.
  (function () {
    var form = document.getElementById("autolabel-form");
    var which = document.getElementById("al-which");
    var field = document.getElementById("al-which-field");
    if (!form || !which || !field || !form.elements.status || !form.elements.auto) return;
    var status = form.elements.status, auto = form.elements.auto;
    var modelPreset = document.getElementById("al-preset");
    var PRESETS = {
      unreviewed: ["unreviewed", ""], reviewed: ["reviewed", ""],
      "reviewed-overlaps": ["reviewed", "overlaps"], "unreviewed-overlaps": ["unreviewed", "overlaps"],
      all: ["any", ""]
    };
    var CUSTOM_ONLY = ["scene", "camera", "label", "frames"];
    var named = function () {
      var other = CUSTOM_ONLY.some(function (n) { var e = form.elements[n]; return e && e.value.trim(); });
      if (other) return "custom";
      var s = status.value || "unreviewed";
      var found = Object.keys(PRESETS).filter(function (k) { return PRESETS[k][0] === s && PRESETS[k][1] === auto.value; });
      return found.length ? found[0] : "custom";
    };
    var syncModel = function () {
      if (!modelPreset) return;
      var s = status.value || "unreviewed";
      modelPreset.value = s === "any" ? "self-training" : s === "unreviewed" ? "first-pass" : "";
    };
    // The line above the button: what a run looks at, while its fields sit under Advanced.
    var scope = function () {
      var line = document.getElementById("al-scope");
      if (!line) return;
      var limit = form.elements.limit ? form.elements.limit.value : "";
      var what = which.value === "custom" ? "frames matching your filters" :
        which.options[which.selectedIndex].text.toLowerCase() + (which.value === "all" ? " frames" : which.value.indexOf("overlaps") > 0 ? "" : " frames");
      var again = form.elements.again && form.elements.again.checked;
      line.textContent = "Looks at up to " + limit + " " + what + (again ? ", also ones these labelers already looked at" : "") + ". Change that under Advanced.";
    };
    var apply = function () {
      var p = PRESETS[which.value];
      if (p) { status.value = p[0]; auto.value = p[1]; }
      form.querySelectorAll("[data-al-custom]").forEach(function (el) { el.hidden = !!p; });
      CUSTOM_ONLY.forEach(function (n) { var e = form.elements[n]; if (e) e.disabled = !!p; });
      syncModel();
      scope();
    };
    which.addEventListener("change", apply);
    form.addEventListener("change", scope);
    status.addEventListener("change", syncModel);
    if (modelPreset) {
      modelPreset.addEventListener("change", function () {
        var all = modelPreset.value === "self-training";
        if (which.value === "custom") { status.value = all ? "any" : "unreviewed"; return; }
        which.value = all ? "all" : "unreviewed";
        apply();
      });
    }
    field.hidden = false;
    which.value = named();
    apply();
  })();

  // Preview frames: what the form would do, on a click. A change to the form clears the answer,
  // which no longer describes it. The classes are the picker's; one not in the class list yet is
  // appended only by a real run, so it is left out here and said so.
  (function () {
    var form = document.getElementById("autolabel-form");
    var run = document.getElementById("al-preview-run");
    var box = document.getElementById("al-preview");
    if (!form || !run || !box) return;
    var seq = 0;
    var line = function (cls, text) {
      var p = document.createElement("p");
      p.className = cls;
      p.textContent = text;
      box.appendChild(p);
    };
    var clear = function () { seq++; box.textContent = ""; box.hidden = true; };
    var draw = function (j, note) {
      box.textContent = "";
      box.hidden = false;
      line("small tr-runs-line", j.summary);
      var warnings = (j.warnings || []).concat(note ? [note] : []);
      if (!warnings.length) return;
      var ul = document.createElement("ul");
      ul.className = "warn-list small";
      warnings.forEach(function (w) {
        var li = document.createElement("li");
        li.textContent = w;
        ul.appendChild(li);
      });
      box.appendChild(ul);
    };
    var fail = function (text) {
      box.textContent = "";
      box.hidden = false;
      line("small error-text", text);
    };
    run.hidden = false;
    form.addEventListener("change", clear);
    run.addEventListener("click", function () {
      var mine = ++seq;
      var body = new URLSearchParams(), note = "";
      Array.prototype.forEach.call(form.elements, function (e) {
        if (!e.name || e.disabled || e.name === "classes") return;
        if ((e.type === "checkbox" || e.type === "radio") && !e.checked) return;
        body.append(e.name, e.value);
      });
      var cls = alClasses ? alClasses() : null;
      if (cls && !cls.names.length) { fail("Choose at least one class to look for."); return; }
      if (cls) {
        var listed = cls.listed();
        if (!listed.length) { fail("Add a picked class to the class list first: the preview counts listed classes only."); return; }
        if (listed.length < cls.names.length) note = "Classes not in the class list yet are left out of this count; a real run adds them first.";
        listed.forEach(function (n) { body.append("classes", n); });
      }
      box.textContent = "";
      box.hidden = false;
      line("small muted", "Counting the frames\u2026");
      run.disabled = true;
      fetch("/training/autolabel/preview", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
        body: body.toString()
      }).then(function (res) { return res.json(); }).then(function (j) {
        if (mine !== seq) return;
        if (j.success) draw(j, note); else fail(j.message || "The server gave no answer.");
      }).catch(function (e) {
        if (mine === seq) fail(e.message);
      }).then(function () { run.disabled = false; });
    });
  })();

  // Auto-label after import: the follow-up options show while the box is ticked (hidden ones
  // are disabled, so not posted), the Review folder switch while "Save frames to" has a value.
  (function () {
    var form = document.getElementById("import-form");
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
      if (!box.checked) return;
      var cls = alClasses ? alClasses() : null;
      if (cls && !cls.names.length) {
        ev.preventDefault();
        say("Auto-label: choose at least one class to look for.");
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
      alOptions().forEach(function (p) { add(p[0], p[1]); });
      if (cls) cls.names.forEach(function (n) { add("classes", n); });
    });
  })();

  var train = document.getElementById("train");
  if (train) {
    // The runs: one per ticked scene per ticked model. The line lists their names before
    // queuing; the model name field works for one run only (the server refuses it with more).
    var name = document.getElementById("train-name");
    var hint = document.getElementById("train-name-hint");
    var runsLine = document.getElementById("train-runs-line");
    var models = train.querySelectorAll('input[name="model"]');
    var sceneBoxes = train.querySelectorAll('input[name="scene"]');
    var balanceBoxes = train.querySelectorAll('input[name="balance"]');
    var hintText = hint ? hint.innerHTML : "";
    var tickedScenes = function () {
      var out = [];
      sceneBoxes.forEach(function (b) { if (b.checked) out.push(b); });
      return out;
    };
    var names = function () {
      var ms = [];
      models.forEach(function (m) { if (m.checked) ms.push(m.value); });
      var sc = tickedScenes();
      // None ticked = all scenes (the server's rule).
      var prefixes = sc.length ? sc.map(function (b) { return b.getAttribute("data-prefix") || "custom"; }) : ["custom"];
      var list = [];
      ms.forEach(function (m) { prefixes.forEach(function (p) { list.push(p + "-" + m); }); });
      var n = list.length;
      var bal = [];
      balanceBoxes.forEach(function (b) { if (b.checked) bal.push(b.value.replace("-", " + ")); });
      if (runsLine) {
        runsLine.hidden = n === 0;
        runsLine.textContent = n === 0 ? "" :
          "Queues " + n + " run" + (n === 1 ? "" : "s") + ": " + list.join(", ") +
          (prefixes.length > 1 ? " (compared on the same held-out frames)" : "") +
          (bal.length ? "; training frames balanced by " + bal.join(", ") : "");
      }
      if (!name) return;
      name.disabled = n > 1;
      if (hint) {
        if (n > 1) hint.textContent = n + " runs: each gets its own name (<scene>-<model>-<date>).";
        else hint.innerHTML = hintText;
      }
    };
    models.forEach(function (m) { m.addEventListener("change", names); });
    sceneBoxes.forEach(function (b) { b.addEventListener("change", names); });
    balanceBoxes.forEach(function (b) { b.addEventListener("change", names); });
    // Night only / IR only ticked: the Other section stays open.
    var other = train.querySelector(".tr-scene-other");
    if (other && other.querySelector("input:checked")) other.open = true;
    names();

    // Learning rate placeholder: the ticked family's default (`data-default-lr` on the nearest
    // element around its checkbox: the size cell or the family row); ticked models of families
    // with different defaults list them all; nothing ticked = the template's text.
    var lrField = document.getElementById("train-lr");
    if (lrField) {
      var lrDefault = lrField.getAttribute("placeholder") || "recommended";
      var lrHint = function () {
        var seen = [];
        models.forEach(function (m) {
          if (!m.checked) return;
          var holder = m.closest("[data-default-lr]");
          var v = holder ? holder.getAttribute("data-default-lr") : "";
          if (v && seen.indexOf(v) < 0) seen.push(v);
        });
        lrField.placeholder = seen.length === 1 ? seen[0] : seen.length > 1 ? "per family: " + seen.join(" / ") : lrDefault;
      };
      models.forEach(function (m) { m.addEventListener("change", lrHint); });
      lrHint();
    }

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

  // Train classes: the class picker (class-picker.js) over the classes of the ticked dataset
  // folders (GET /v1/training/dataset-classes), each with its boxes per scene and warning chips
  // (few boxes; none in a ticked scene). Starts with the last choice (localStorage), else what
  // the enabled models detect (the auto-label default), else every class with boxes. The picked
  // classes post as repeated `classes` fields, in the folders' class-list order.
  (function () {
    var form = document.getElementById("train"), CP = window.ClassPicker;
    var holder = document.getElementById("train-picker"), msg = document.getElementById("train-classes-msg");
    if (!form || !holder || !CP) return;
    var KEY = "bop.training.classes";
    var FEW = 50;
    var picker = null, inv = null, byName = {}, catalog = null, seq = 0, timer = null, lastQuery = null;
    var mk = function (tag, cls, text) {
      var e = document.createElement(tag);
      if (cls) e.className = cls;
      if (text != null) e.textContent = text;
      return e;
    };
    var note = function (text) { holder.replaceChildren(mk("p", "muted small", text)); };
    // The balance preview follows the picked classes.
    var changed = function () { form.dispatchEvent(new Event("tr-classes")); };
    trainClasses = function () { return picker ? inOrder(picker.getSelected()) : null; };
    var stored = function () {
      try {
        var v = JSON.parse(window.localStorage.getItem(KEY) || "null");
        return Array.isArray(v) ? v.filter(function (x) { return typeof x === "string"; }) : null;
      } catch (e) { return null; }
    };
    var store = function (names) {
      try { window.localStorage.setItem(KEY, JSON.stringify(names)); } catch (e) { /* storage blocked */ }
    };
    var datasets = function () {
      var out = [];
      form.querySelectorAll('input[name="dataset"]').forEach(function (b) { if (b.checked && !b.disabled) out.push(b.value); });
      var extra = form.elements.dataset_path;
      if (extra && extra.value.trim()) out.push(extra.value.trim());
      return out;
    };
    var scenes = function () {
      var out = [];
      form.querySelectorAll('input[name="scene"]').forEach(function (b) { if (b.checked) out.push(b.value); });
      return out.length ? out : ["all"];
    };
    var sceneText = function (v) { return v === "all" ? "all scenes" : v.split("+").map(function (k) { return k === "ir" ? "IR" : k; }).join(" + "); };
    // Boxes of class count `c` in the scene filter `v` (`all`, `day`, `night+ir`, ...).
    var boxesIn = function (c, v) {
      if (v === "all") return c.boxes;
      return v.split("+").reduce(function (n, k) { return n + ((c.by_scene || {})[k] || 0); }, 0);
    };
    var present = function () {
      var withBoxes = (inv.classes || []).filter(function (c) { return c.boxes > 0; }).map(function (c) { return c.name; });
      return withBoxes.length ? withBoxes : (inv.classes || []).map(function (c) { return c.name; });
    };
    var inOrder = function (names) {
      var order = inv.order || [];
      return names.slice().sort(function (a, b) { return order.indexOf(a) - order.indexOf(b); });
    };
    var only = function (names, within) { return (names || []).filter(function (n) { return within.indexOf(n) >= 0; }); };
    var defaults = function () {
      var p = present();
      var last = only(stored(), p);
      if (last.length) return last;
      var rel = only((catalog && catalog.relevant) || [], p);
      return rel.length ? rel : p;
    };
    var extra = function (entry, chip) {
      var c = byName[entry.name];
      if (!c) return;
      var label = chip.querySelector(".cp-label") || chip;
      var s = c.by_scene || {};
      var count = mk("span", "tr-cls-count", c.boxes + " · day " + (s.day || 0) + " / night " + (s.night || 0) + " / IR " + (s.ir || 0));
      count.title = c.boxes + " boxes in " + c.frames + " frames: " + (s.day || 0) + " day, " + (s.night || 0) + " night, " +
        (s.ir || 0) + " IR" + (s.unknown ? ", " + s.unknown + " without a scene tag" : "");
      label.appendChild(count);
      if (c.boxes < FEW) {
        var few = mk("span", "cp-badge tr-cls-warn", "few boxes: under " + FEW);
        few.title = "Under " + FEW + " boxes is thin for fine-tuning; label more frames of it, or leave it out.";
        label.appendChild(few);
      }
      scenes().forEach(function (v) {
        if (v === "all" || boxesIn(c, v) > 0) return;
        var none = mk("span", "cp-badge tr-cls-warn tr-scene-warn", "no " + sceneText(v) + " boxes");
        none.title = "The " + sceneText(v) + " run has no box of " + c.name + " to learn from.";
        label.appendChild(none);
      });
    };
    var say = function () {
      changed();
      if (!msg) return;
      msg.className = "small";
      if (!inv) { msg.textContent = ""; return; }
      var parts = [];
      var f = inv.frames || {};
      var total = (f.day || 0) + (f.night || 0) + (f.ir || 0) + (f.unknown || 0);
      if (total) {
        parts.push(total + " frames: " + (f.day || 0) + " day, " + (f.night || 0) + " night, " + (f.ir || 0) + " IR" +
          (f.unknown ? ", " + f.unknown + " without a scene tag (only in all-scene runs)" : ""));
      }
      var names = picker ? picker.getSelected() : [];
      var warn = [];
      names.forEach(function (n) {
        var c = byName[n];
        if (!c) return;
        scenes().forEach(function (v) {
          if (v !== "all" && boxesIn(c, v) === 0) warn.push(n + " has no " + sceneText(v) + " boxes");
        });
      });
      var errs = (inv.errors || []).map(function (e) { return e.dataset + ": " + e.message; });
      if (picker && !names.length) warn.push("no class ticked");
      msg.textContent = parts.concat(warn, errs).join(". ") + (parts.length || warn.length || errs.length ? "." : "");
      if (warn.length || errs.length) msg.className = "small error-text";
    };
    // The picker's entries: the folders' classes in list order, grouped by the catalog's category.
    var entriesNow = function () {
      var cats = {};
      ((catalog && catalog.entries) || []).forEach(function (e) { cats[e.name] = e.category; });
      return (inv.order || []).filter(function (n) { return byName[n]; }).map(function (n) {
        return { name: n, category: cats[n] || "object", in_list: true, coco: false, models: [] };
      });
    };
    var build = function () {
      byName = {};
      (inv.classes || []).forEach(function (c) { byName[c.name] = c; });
      var entries = entriesNow();
      if (!entries.length) {
        picker = null;
        note((inv.errors || []).length ? "No classes could be read from the ticked folders." : "The ticked folders have no classes.");
        say();
        return;
      }
      var p = present();
      var keep = picker ? only(picker.getSelected(), p) : [];
      var sel = keep.length ? keep : defaults();
      var rel = only((catalog && catalog.relevant) || [], p);
      var quick = [{ label: "All with boxes", names: p }];
      if (rel.length) quick.push({ label: "What your models detect", names: rel });
      picker = CP.create(holder, {
        entries: entries, selected: sel, mode: "compact", label: "Classes", emptyText: "none",
        quickSets: quick, extra: extra,
        onChange: function (names) { store(names); say(); }
      });
      picker.el.open = true;
      say();
    };
    var load = function () {
      var list = datasets();
      var q = list.map(function (d) { return "dataset=" + encodeURIComponent(d); }).join("&");
      if (q === lastQuery) return;
      lastQuery = q;
      var mine = ++seq;
      if (!list.length) {
        inv = null; picker = null;
        note("Tick a dataset folder to list its classes; none ticked = every class of the export.");
        say();
        return;
      }
      if (!picker) note("Reading the classes of the ticked folders…");
      var cat = catalog ? Promise.resolve(catalog) : CP.fetchCatalog().catch(function () { return {}; });
      Promise.all([
        cat,
        fetch("/v1/training/dataset-classes?" + q, { headers: { Accept: "application/json" } })
          .then(function (res) { return res.json(); })
      ]).then(function (r) {
        if (mine !== seq) return;
        catalog = r[0] || {};
        if (!r[1] || r[1].success === false) throw new Error((r[1] && r[1].message) || "no answer");
        inv = r[1];
        build();
      }).catch(function (e) {
        if (mine !== seq) return;
        inv = null; picker = null; lastQuery = null;
        note("The classes could not be read (" + e.message + "); every class of the export is used.");
        say();
      });
    };
    var later = function () {
      if (timer) clearTimeout(timer);
      timer = setTimeout(load, 500);
    };
    form.querySelectorAll('input[name="dataset"]').forEach(function (b) { b.addEventListener("change", load); });
    var extraPath = form.elements.dataset_path;
    if (extraPath) { extraPath.addEventListener("input", later); extraPath.addEventListener("change", later); }
    // The scene warnings follow the ticked scenes.
    form.querySelectorAll('input[name="scene"]').forEach(function (b) {
      b.addEventListener("change", function () {
        if (picker && inv) picker.setEntries(entriesNow());
        say();
      });
    });
    form.addEventListener("submit", function (ev) {
      form.querySelectorAll("input.tr-cls-field").forEach(function (i) { i.remove(); });
      // Save folder / Remove post to /training/datasets: no class check.
      var by = ev.submitter;
      if (by && by.getAttribute("formaction")) return;
      if (!picker) return;
      var names = inOrder(picker.getSelected());
      if (!names.length) {
        ev.preventDefault();
        if (msg) { msg.textContent = "Choose at least one class to train."; msg.className = "small error-text"; }
        return;
      }
      names.forEach(function (n) {
        var i = document.createElement("input");
        i.type = "hidden"; i.name = "classes"; i.value = n; i.className = "tr-cls-field";
        form.appendChild(i);
      });
    });
    load();
  })();

  // Train balance: the fieldset's state in its summary and the preview of what balancing does
  // to each run's training frames (POST /v1/training/balance-preview with the form's dataset,
  // scene, class, excluded, public and balance fields; debounced, a stale answer is aborted or
  // dropped), shown only while an axis is ticked: images per epoch before -> after, then boxes
  // per class, frames per scene, the top cameras (and camera-days with that axis) with bars,
  // Several scenes ticked: one tab per run. The same answer draws the pre-flight report above Queue
  // training, always: boxes per class and scene, the splits, what was left out, the disk, the
  // warnings and what would refuse the run.
  (function () {
    var form = document.getElementById("train");
    var box = document.getElementById("train-balance");
    var panel = document.getElementById("train-balance-preview");
    var pre = document.getElementById("train-preflight");
    var stateEl = document.getElementById("train-balance-state");
    if (!form || !box || !panel) return;
    var TOP = 8;
    var LABEL = { "class": "class", scene: "scene", camera: "camera", "camera-day": "camera + day" };
    var timer = null, ctrl = null, seq = 0, lastBody = null, tab = 0;
    var capOn = document.getElementById("train-bal-cap-on"), cap = document.getElementById("train-bal-cap");
    var mk = function (tag, cls, text) {
      var e = document.createElement(tag);
      if (cls) e.className = cls;
      if (text != null) e.textContent = text;
      return e;
    };
    var axes = function () {
      var out = [];
      form.querySelectorAll('input[name="balance"]').forEach(function (b) { if (b.checked) out.push(b.value); });
      return out;
    };
    // The cap's number posts (and is editable) only while its box is ticked.
    var syncCap = function () { if (cap && capOn) cap.disabled = !capOn.checked; };
    // The fields the preview needs, like the queue form posts them (classes from the picker).
    var body = function () {
      var p = new URLSearchParams();
      form.querySelectorAll('input[name="dataset"]').forEach(function (b) { if (b.checked && !b.disabled) p.append("dataset", b.value); });
      var extra = form.elements.dataset_path;
      if (extra && extra.value.trim()) p.append("dataset_path", extra.value.trim());
      form.querySelectorAll('input[name="scene"]:checked').forEach(function (b) { p.append("scene", b.value); });
      var cls = trainClasses ? trainClasses() : null;
      if (cls) cls.forEach(function (n) { p.append("classes", n); });
      else {
        var t = form.querySelector('#train-picker input[name="classes"]');
        if (t && t.value.trim()) p.append("classes", t.value.trim());
      }
      ["excluded", "balance_strength"].forEach(function (n) {
        var r = form.querySelector('input[name="' + n + '"]:checked');
        if (r) p.append(n, r.value);
      });
      var pub = form.querySelector('input[name="public"]');
      if (pub && pub.checked && !pub.disabled) p.append("public", "on");
      axes().forEach(function (a) { p.append("balance", a); });
      var max = form.elements.balance_max_repeat;
      if (max && max.value !== "") p.append("balance_max_repeat", max.value);
      if (capOn && capOn.checked) {
        p.append("balance_cap_on", "on");
        if (cap && cap.value !== "") p.append("balance_cap", cap.value);
      }
      // The disk estimate of the ticked models.
      form.querySelectorAll('input[name="model"]:checked').forEach(function (b) { p.append("model", b.value); });
      var seeds = form.elements.seeds, dir = form.elements.install_dir;
      if (seeds && seeds.value) p.append("seeds", seeds.value);
      if (dir && dir.value.trim()) p.append("install_dir", dir.value.trim());
      return p.toString();
    };
    var say = function (text, cls) {
      panel.replaceChildren(mk("p", cls || "muted small", text));
    };
    var pct = function (n, max) { return max > 0 ? Math.max(n > 0 ? 1 : 0, Math.round(1000 * n / max) / 10) : 0; };
    // One before -> after table: rows [name, before, after], bars scaled to the table's largest.
    var table = function (title, head, rows) {
      var wrap = mk("div", "tr-bal-table");
      wrap.appendChild(mk("h4", "tr-bal-h", title));
      if (!rows.length) { wrap.appendChild(mk("p", "muted small", "none")); return wrap; }
      var max = 0;
      rows.forEach(function (r) { max = Math.max(max, r[1], r[2]); });
      var t = mk("table", "tr-btable");
      var tr = mk("tr");
      [[head, ""], ["Before", "num"], ["After", "num"], ["", "tr-bal-bars"]].forEach(function (h) {
        var th = mk("th", h[1] || null, h[0]);
        th.scope = "col";
        tr.appendChild(th);
      });
      var thead = mk("thead");
      thead.appendChild(tr);
      t.appendChild(thead);
      var tb = mk("tbody");
      rows.forEach(function (r) {
        var row = mk("tr");
        var name = mk("td", "tr-bal-name", r[0]);
        name.title = r[0];
        row.appendChild(name);
        row.appendChild(mk("td", "num", String(r[1])));
        var after = mk("td", "num" + (r[2] > r[1] ? " tr-bal-up" : r[2] < r[1] ? " tr-bal-down" : ""), String(r[2]));
        if (r[1] > 0 && r[2] !== r[1]) after.title = "x" + (r[2] / r[1]).toFixed(2);
        row.appendChild(after);
        var bars = mk("td", "tr-bal-bars");
        var b1 = mk("span", "tr-bal-bar tr-bal-before"), b2 = mk("span", "tr-bal-bar tr-bal-after");
        b1.style.width = pct(r[1], max) + "%";
        b2.style.width = pct(r[2], max) + "%";
        bars.appendChild(b1);
        bars.appendChild(b2);
        row.appendChild(bars);
        tb.appendChild(row);
      });
      t.appendChild(tb);
      wrap.appendChild(t);
      return wrap;
    };
    // Rows of two {name: n} maps: every name, the largest before first; at most `top`.
    var rowsOf = function (before, after, top) {
      var names = Object.keys(before || {});
      Object.keys(after || {}).forEach(function (k) { if (names.indexOf(k) < 0) names.push(k); });
      var rows = names.map(function (k) { return [k, (before || {})[k] || 0, (after || {})[k] || 0]; });
      rows.sort(function (a, b) { return b[1] - a[1] || b[2] - a[2] || (a[0] < b[0] ? -1 : 1); });
      return top ? rows.slice(0, top) : rows;
    };
    var share = function (x) { return Math.round(100 * (x || 0)) + "%"; };
    var runPane = function (r) {
      var d = mk("div", "tr-bal-run");
      if (r.error) { d.appendChild(mk("p", "error-text small", r.error)); return d; }
      var rep = r.report;
      var counts = r.train_frames + " training frames, " + r.valid_frames + " validation, " + r.heldout_frames + " held out (never balanced).";
      if (!rep) {
        d.appendChild(mk("p", "muted small", counts + " Nothing to balance."));
      } else {
        var b = rep.before || {}, a = rep.after || {};
        var kpi = mk("p", "tr-bal-kpi");
        kpi.appendChild(mk("span", "k", "Images per epoch "));
        kpi.appendChild(mk("strong", null, b.images + " → " + a.images));
        if (b.images > 0) kpi.appendChild(mk("span", "muted small", " (×" + (a.images / b.images).toFixed(2) + ")"));
        d.appendChild(kpi);
        var facts = counts + " Largest repeat " + rep.max_repeat + "×; frames without a box " + share(b.negative_share) + " → " + share(a.negative_share) + ".";
        d.appendChild(mk("p", "muted small", facts));
        if (rep.dropped > 0) {
          d.appendChild(mk("p", "small tr-bal-drop", "The cap leaves out " + rep.dropped + " training frame" + (rep.dropped === 1 ? "" : "s") +
            " of dominant groups (" + rep.frames_before + " → " + rep.frames_after + " unique frames)."));
        }
        var grid = mk("div", "tr-bal-tables");
        grid.appendChild(table("Boxes per class", "Class", rowsOf(b.boxes_per_class, a.boxes_per_class, 0)));
        grid.appendChild(table("Frames per scene", "Scene", rowsOf(b.frames_per_scene, a.frames_per_scene, 0)));
        var cams = rowsOf(b.frames_per_camera, a.frames_per_camera, TOP);
        var nCams = Object.keys(b.frames_per_camera || {}).length;
        grid.appendChild(table(nCams > TOP ? "Top " + TOP + " of " + nCams + " cameras" : "Frames per camera", "Camera", cams));
        if (((rep.balance || {}).axes || []).indexOf("camera-day") >= 0) {
          var after = {};
          (a.camera_days || []).forEach(function (x) { after[x[0]] = x[1]; });
          var days = (b.camera_days || []).slice(0, TOP).map(function (x) { return [x[0], x[1], after[x[0]] || 0]; });
          grid.appendChild(table("Top camera-days", "Camera / day", days));
        }
        d.appendChild(grid);
      }
      return d;
    };
    var list = function (items, cls) {
      var ul = mk("ul", cls);
      items.forEach(function (x) { ul.appendChild(mk("li", null, x)); });
      return ul;
    };
    // One run of the pre-flight report: the counts, then in plain sentences what to do about them.
    var preRun = function (r, several) {
      var d = mk("div", "tr-pre-run");
      if (several) d.appendChild(mk("h4", "tr-bal-h", r.title || r.scene));
      if (r.error) { d.appendChild(mk("p", "error-text small", "This run would not start: " + r.error)); return d; }
      var p = r.preflight || {};
      (p.refusals || []).forEach(function (x) { d.appendChild(mk("p", "error-text small", "This run would not start: " + x)); });
      d.appendChild(mk("p", "small", p.train_frames + " training frames (" + p.train_images + " images per epoch, " + p.negatives +
        " without objects), " + p.valid_frames + " validation, " + p.heldout_frames + " held out for the benchmark."));
      var sc = p.scenes || {};
      var parts = [["day", sc.day], ["night", sc.night], ["IR", sc.ir], ["no scene tag", sc.unknown]]
        .filter(function (x) { return x[1] > 0; })
        .map(function (x) { return x[1] + " " + x[0]; });
      if (parts.length) d.appendChild(mk("p", "muted small", "Training frames by scene: " + parts.join(", ") + "."));
      var rows = p.classes || [];
      if (rows.length) {
        var t = mk("table", "tr-btable tr-pre-table");
        var head = mk("tr");
        ["Boxes per class", "Train", "Val", "Held out", "Day", "Night", "IR"].forEach(function (h, i) {
          var th = mk("th", i ? "num" : null, h);
          th.scope = "col";
          head.appendChild(th);
        });
        var thead = mk("thead");
        thead.appendChild(head);
        t.appendChild(thead);
        var tb = mk("tbody");
        rows.forEach(function (c) {
          var tr = mk("tr");
          var s = c.by_scene || {};
          tr.appendChild(mk("td", "tr-bal-name", c.name));
          [c.train, c.valid, c.heldout, s.day, s.night, s.ir].forEach(function (n) { tr.appendChild(mk("td", "num", String(n || 0))); });
          tb.appendChild(tr);
        });
        t.appendChild(tb);
        var wrap = mk("div", "scroll");
        wrap.appendChild(t);
        d.appendChild(wrap);
      }
      var left = [
        [p.ignore_frames, "with unsure or too-small boxes"],
        [p.parked_capped, "over the parked-spot cap"],
        [p.duplicates_dropped, "that repeat a frame of another export"],
        [p.scene_filtered, "outside this run's scenes"]
      ].filter(function (x) { return x[0] > 0; }).map(function (x) { return x[0] + " " + x[1]; });
      if (left.length) d.appendChild(mk("p", "muted small", "Left out of training: " + left.join("; ") + "."));
      if (p.hard_negatives > 0) d.appendChild(mk("p", "muted small", p.hard_negatives + " of the frames without objects were confirmed empty or had a box rejected on the Review tab."));
      if ((p.notes || []).length) d.appendChild(list(p.notes, "muted small"));
      if ((p.warnings || []).length) d.appendChild(list(p.warnings, "warn-list small"));
      return d;
    };
    var renderPre = function (j) {
      if (!pre) return;
      var runs = j.runs || [];
      pre.replaceChildren(mk("h3", "tr-bal-h", "Before you train"));
      runs.forEach(function (r) { pre.appendChild(preRun(r, runs.length > 1)); });
      var disk = j.disk || {};
      (disk.lines || []).forEach(function (l) { pre.appendChild(mk("p", "muted small", "Disk: " + l.text + ".")); });
      if (disk.refusal) pre.appendChild(mk("p", "error-text small", disk.refusal));
      pre.hidden = false;
    };
    var render = function (j) {
      var runs = j.runs || [];
      panel.replaceChildren();
      if (!runs.length) { say("No run to preview."); return; }
      var legend = mk("p", "small tr-bal-legend");
      legend.appendChild(mk("span", "tr-bal-swatch tr-bal-before"));
      legend.appendChild(document.createTextNode(" before "));
      legend.appendChild(mk("span", "tr-bal-swatch tr-bal-after"));
      legend.appendChild(document.createTextNode(" after balancing (repeats counted; per epoch)"));
      if (tab >= runs.length) tab = 0;
      var panes = runs.map(runPane);
      if (runs.length > 1) {
        var bar = mk("div", "tr-bal-tabs");
        bar.setAttribute("role", "tablist");
        var buttons = runs.map(function (r, i) {
          var b = mk("button", "small" + (i === tab ? "" : " secondary"), r.title || r.scene);
          b.type = "button";
          b.setAttribute("role", "tab");
          b.setAttribute("aria-selected", i === tab ? "true" : "false");
          b.addEventListener("click", function () {
            tab = i;
            buttons.forEach(function (x, k) {
              x.setAttribute("aria-selected", k === i ? "true" : "false");
              x.className = "small" + (k === i ? "" : " secondary");
            });
            panes.forEach(function (p, k) { p.hidden = k !== i; });
          });
          bar.appendChild(b);
          return b;
        });
        panel.appendChild(bar);
      }
      panel.appendChild(legend);
      panes.forEach(function (p, i) { p.hidden = i !== tab; p.setAttribute("role", runs.length > 1 ? "tabpanel" : "region"); panel.appendChild(p); });
    };
    var anyData = function () {
      var extra = form.elements.dataset_path;
      return !!form.querySelector('input[name="dataset"]:checked:not(:disabled)') || !!(extra && extra.value.trim());
    };
    var run = function () {
      timer = null;
      var ax = axes();
      if (stateEl) stateEl.textContent = ax.length ? "on: " + ax.map(function (a) { return LABEL[a] || a; }).join(" + ") : "off";
      if (!ax.length) {
        panel.hidden = true;
        panel.replaceChildren();
      }
      var stop = function () {
        if (ctrl) ctrl.abort();
        ctrl = null; seq++; lastBody = null;
      };
      if (!anyData()) {
        stop();
        if (pre) { pre.hidden = true; pre.replaceChildren(); }
        return;
      }
      var cls = trainClasses ? trainClasses() : null;
      if (cls && !cls.length) {
        stop();
        if (pre) { pre.hidden = false; pre.replaceChildren(mk("p", "small error-text", "Tick at least one class.")); }
        if (ax.length) { panel.hidden = false; say("Tick at least one class to preview the balance.", "small error-text"); }
        return;
      }
      var b = body();
      if (b === lastBody) return;
      lastBody = b;
      if (ctrl) ctrl.abort();
      ctrl = typeof AbortController === "function" ? new AbortController() : null;
      var mine = ++seq;
      if (ax.length) {
        panel.hidden = false;
        panel.classList.add("tr-bal-busy");
        if (!panel.firstChild) say("Previewing the balance…");
      }
      fetch("/v1/training/balance-preview", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
        body: b,
        signal: ctrl ? ctrl.signal : undefined
      }).then(function (res) {
        return res.json().catch(function () { return null; }).then(function (j) {
          if (!res.ok || !j || j.success === false) throw new Error((j && j.message) || "the server answered HTTP " + res.status);
          return j;
        });
      }).then(function (j) {
        if (mine !== seq) return;
        panel.classList.remove("tr-bal-busy");
        renderPre(j);
        if (axes().length) render(j);
      }).catch(function (e) {
        if (mine !== seq || (e && e.name === "AbortError")) return;
        panel.classList.remove("tr-bal-busy");
        lastBody = null;
        if (axes().length) say("No preview: " + e.message, "small error-text");
        if (pre) { pre.hidden = false; pre.replaceChildren(mk("p", "small error-text", "No report: " + e.message)); }
      });
    };
    var later = function () {
      if (timer) clearTimeout(timer);
      timer = setTimeout(run, 400);
    };
    // Every field of the form (the Datasets step, and models, seeds and the install folder for the
    // disk estimate); the class picker's changes come as tr-classes. An unchanged form asks nothing.
    form.addEventListener("change", later);
    form.addEventListener("input", later);
    form.addEventListener("tr-classes", later);
    if (capOn) capOn.addEventListener("change", syncCap);
    syncCap();
    later();
  })();

  // A lost class list can hold 70 COCO names: the ones a camera triggers on first, then a count.
  var CCTV_CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle", "motorbike", "dog", "cat", "bird", "horse", "bear", "deer", "fox", "raccoon"];
  var shortClasses = function (list) {
    var first = list.filter(function (c) { return CCTV_CLASSES.indexOf(c) >= 0; });
    var shown = first.length ? first : list.slice(0, 8);
    var more = list.length - shown.length;
    return shown.join(", ") + (more > 0 ? " and " + more + " more" : "");
  };

  // Models tab, Use card, "Use as default": GET /v1/training/switch-preview draws what would change in the
  // row under the model, nothing is saved until Switch is pressed. Copy settings changes the
  // "After" column; the threshold box too. Without JS the form posts a switch with the settings
  // copied, which Revert undoes.
  (function () {
    var mk = function (tag, cls, text) {
      var e = document.createElement(tag);
      if (cls) e.className = cls;
      if (text != null) e.textContent = text;
      return e;
    };
    var hidden = function (name, value) {
      var i = mk("input"); i.type = "hidden"; i.name = name; i.value = value; return i;
    };
    var list = function (cls, lines) {
      var ul = mk("ul", cls);
      lines.forEach(function (t) { ul.appendChild(mk("li", null, t)); });
      return ul;
    };
    var checkbox = function (name, on, text) {
      var l = mk("label", "check"), i = mk("input");
      i.type = "checkbox"; i.name = name; i.value = "1"; i.checked = on;
      l.appendChild(i); l.appendChild(document.createTextNode(" " + text));
      return { label: l, input: i };
    };
    function draw(host, id, j, close) {
      var already = j.already_default;
      var nodes = [mk("h3", null, already ? j.new + " is the default model" : "Switch the default model: " + j.current + " to " + j.new)];
      if (j.compare) nodes.push(mk("p", null, j.compare));
      if (j.trained_on) nodes.push(mk("p", "warn", j.trained_on + ": its scores above look better than it will do on new frames."));
      var form = mk("form");
      form.method = "post"; form.action = "/training/use";
      form.appendChild(hidden("id", id));
      form.appendChild(hidden("role", "default"));
      form.appendChild(hidden("expect_current", j.current));
      var cls = j.classes, copy = null, use = null;
      if (!already) {
        if (cls.lost.length) nodes.push(mk("p", "warn", "Classes lost: " + shortClasses(cls.lost) + ". " + j.current + " reports them today and " + j.new + " will not. A Blue Iris camera that triggers on these labels stops alerting for them, unless you change its AI settings."));
        if (cls.unknown) nodes.push(mk("p", "warn", "The classes could not be compared: " + cls.unknown + "."));
        var gained = mk("p", "muted small");
        nodes.push(gained);
        var table = mk("table"), head = mk("tr"), body = mk("tbody");
        ["Setting", "Now (" + j.current + ")", "After (" + j.new + ")"].forEach(function (t) { head.appendChild(mk("th", null, t)); });
        var thead = mk("thead"); thead.appendChild(head); table.appendChild(thead); table.appendChild(body);
        var after = [];
        j.rows.forEach(function (r) {
          var tr = mk("tr"), cell = mk("td");
          tr.appendChild(mk("td", null, r.label));
          tr.appendChild(mk("td", null, r.current));
          tr.appendChild(cell);
          body.appendChild(tr);
          after.push({ row: r, cell: cell });
        });
        var scroll = mk("div", "scroll"); scroll.appendChild(table);
        nodes.push(scroll);
        var classRow = mk("p", "small", "Classes now: " + cls.current + ". After: " + cls.new + ".");
        nodes.push(classRow);
        copy = checkbox("copy", true, "Copy settings from " + j.current);
        form.appendChild(copy.label);
        var copied = list("small", j.copy.copied.map(function (c) { return c.label + ": " + c.value; }));
        var left = list("muted small", j.copy.not_copied.map(function (c) { return c.label + " not copied: " + c.why; }));
        if (j.copy.copied.length) form.appendChild(copied);
        form.appendChild(left);
      }
      var s = j.suggestion;
      if (s) {
        if (already) {
          form.appendChild(hidden("use_threshold", "1"));
          form.appendChild(mk("p", null, "Suggested threshold " + s.threshold.toFixed(2) + ". " + s.words));
        } else {
          use = checkbox("use_threshold", false, "Use the suggested threshold " + s.threshold.toFixed(2));
          form.appendChild(use.label);
          form.appendChild(mk("p", "small", s.words));
        }
        if (s.notes.length) form.appendChild(list("muted small", s.notes));
      } else if (j.no_suggestion) {
        form.appendChild(mk("p", "muted small", "No threshold to suggest: " + j.no_suggestion + "."));
      }
      if (!already) form.appendChild(mk("p", "muted small", "A Blue Iris camera that names a custom model in its AI settings keeps using that model; only cameras with an empty custom models field follow this switch. " + j.current + " stays loaded, and Revert goes back to it."));
      var bar = mk("div", "actions"), go = mk("button", null, already ? (s ? "Use " + s.threshold.toFixed(2) : "Nothing to change") : "Switch to " + j.new);
      go.type = "submit"; go.disabled = already && !s;
      var cancel = mk("button", "secondary", "Close");
      cancel.type = "button"; cancel.addEventListener("click", close);
      bar.appendChild(go); bar.appendChild(cancel); form.appendChild(bar);
      nodes.push(form);
      host.replaceChildren.apply(host, nodes);
      if (already) return;
      var refresh = function () {
        var on = copy.input.checked, g = on ? cls.gained_with_copy : cls.gained;
        after.forEach(function (a) {
          var t = on ? a.row.with_copy : a.row.new;
          if (use && use.input.checked && a.row.key === "confidence_threshold") t = s.threshold.toFixed(2) + " (suggested)";
          a.cell.textContent = t;
        });
        gained.textContent = g.length ? "New classes it will report: " + g.join(", ") + "." : "";
        gained.hidden = !g.length;
        copied.hidden = !on; left.hidden = !on;
      };
      copy.input.addEventListener("change", refresh);
      if (use) use.input.addEventListener("change", refresh);
      refresh();
    }
    document.querySelectorAll("form[data-switch]").forEach(function (f) {
      var id = f.dataset["switch"];
      var host = document.querySelector('.tr-switch[data-run="' + CSS.escape(id) + '"]');
      if (!host) return;
      var row = host.closest("tr");
      var close = function () { row.hidden = true; host.replaceChildren(); };
      f.addEventListener("submit", function (e) {
        e.preventDefault();
        if (!row.hidden) { close(); return; }
        row.hidden = false;
        host.replaceChildren(mk("p", "muted small", "Reading what would change..."));
        fetch("/v1/training/switch-preview?id=" + encodeURIComponent(id), { cache: "no-store" })
          .then(function (r) {
            return r.json().then(function (j) {
              if (!r.ok) throw new Error(j.error || "HTTP " + r.status);
              return j;
            });
          })
          .then(function (j) { draw(host, id, j, close); })
          .catch(function (err) { host.replaceChildren(mk("p", "error-text", String(err.message || err))); });
      });
    });
    // Replacing a verifier asks first, naming it; the server refuses without that answer.
    var card = document.getElementById("use"), verifiers = {};
    try { verifiers = JSON.parse((card && card.dataset.verifiers) || "{}"); } catch (e) { verifiers = {}; }
    document.querySelectorAll("form[data-use-as]").forEach(function (f) {
      f.addEventListener("submit", function (e) {
        if (f.elements.role.value !== "verify") return;
        var target = f.elements.target.value, existing = verifiers[target], name = f.dataset.useAs;
        if (!existing || existing.toLowerCase() === name.toLowerCase()) return;
        if (window.confirm(existing + " already verifies " + target + ". Replace it with " + name + "? Its verifier settings go back to the defaults.")) {
          f.elements.replace_verifier.value = existing;
        } else {
          e.preventDefault();
        }
      });
    });
  })();

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
    if (p.select_fallback) out.push("selected by mAP (" + metricShort(p.select_fallback.metric) + " failed at epoch " + p.select_fallback.epoch + ")");
    else if (select && select !== "map") out.push("selected by " + metricShort(select));
    var seed = s && isNum(s.seed) ? s.seed : r.request && isNum(r.request.train_seed) ? r.request.train_seed : null;
    if (seed !== null) out.push("seed " + seed);
    if (s && typeof s.scene === "string" && s.scene && s.scene !== "all") out.push(s.scene);
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
  // The light answer of GET /v1/training?poll=1 (src/server/training/poll.rs): the epochs this page
  // holds are named in `have` and not sent again; TrainMetrics.poll puts the whole history back on
  // each run, so the answer reads as a full one. No capability (the GPU-fit block above fetches
  // it once), no losses, logs only of busy jobs, no metrics catalog after the first (renderMetrics
  // keeps it). Without train-metrics.js (no run on the page)
  // nothing is held and every history comes whole.
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
      // An early-stop point passed while the training loss still falls: Stop now / Continue.
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
      if (!(a || b)) return true;
      // Not reload(): the page may be the answer to a POST, and reloading it would ask to
      // resubmit (and queue the job again). The poll navigates, not the user: no activity.
      if (window.PrismPoll && window.PrismPoll.navigate) window.PrismPoll.navigate(window.location.pathname + window.location.hash);
      else window.location.assign(window.location.pathname + window.location.hash);
      return false;
    });
  }
  if (window.PrismPoll) {
    window.PrismPoll.create({ run: tick, period: 3000, more: function (go) { return go !== false; } }).start();
  } else {
    // An older live.js: the same loop without the back-off.
    (function () {
      var timer = null, inFlight = false, done = false;
      var go = function () {
        if (timer !== null) { clearTimeout(timer); timer = null; }
        if (done || inFlight || document.hidden) return;
        inFlight = true;
        tick().then(function (more) { done = more === false; }, function () {}).then(function () {
          inFlight = false;
          if (!done && !document.hidden) timer = setTimeout(go, 3000);
        });
      };
      document.addEventListener("visibilitychange", function () {
        if (document.hidden) { if (timer !== null) { clearTimeout(timer); timer = null; } } else go();
      });
      go();
    })();
  }
})();
