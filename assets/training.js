// Training page: the forms' show / hide (import source, auto-label mode, night augmentation,
// the model name with several models ticked; without JS every field shows and works), live
// progress of the queued and running training runs and import / auto-label jobs (polls
// GET /v1/training) and a reload when one finishes, so the results show.
// Auto-label after import: a ticked import posts the Auto-label card's fields as `al.<name>`
// hidden inputs (its classes included); the box and its follow-up options are remembered in
// localStorage. A Recent imports row's Auto-label button fills the card's filter (import=<id>).
// Train: the runs line (one run per ticked scene per model, named before queuing), the class
// picker of the ticked folders (counts per scene, warnings), the balance preview (what the
// Balance fieldset does to each run's training frames) and Finish now while training, plus the
// Stop now / Continue notice of an early-stop point passed while the training loss still falls.
// Each run row gets a settings line (device, batch, input size, learning rate and schedule,
// patience, seed, ...) and the Learning rate field's placeholder follows the ticked family.
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
        entries: entriesOf(cat), selected: restoredClasses["autolabel-form"] || CP.util.defaultSelection(cat), mode: "full", allowAdd: true,
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
  // and the warnings. Several scenes ticked: one tab per run.
  (function () {
    var form = document.getElementById("train");
    var box = document.getElementById("train-balance");
    var panel = document.getElementById("train-balance-preview");
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
      var w = r.warnings || [];
      if (w.length) {
        var ul = mk("ul", "warn-list small");
        w.forEach(function (x) { ul.appendChild(mk("li", null, x)); });
        d.appendChild(ul);
      }
      return d;
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
    var run = function () {
      timer = null;
      var ax = axes();
      if (stateEl) stateEl.textContent = ax.length ? "on: " + ax.map(function (a) { return LABEL[a] || a; }).join(" + ") : "off";
      if (!ax.length) {
        if (ctrl) ctrl.abort();
        ctrl = null; seq++; lastBody = null;
        panel.hidden = true;
        panel.replaceChildren();
        return;
      }
      var cls = trainClasses ? trainClasses() : null;
      if (cls && !cls.length) {
        lastBody = null;
        panel.hidden = false;
        say("Tick at least one class to preview the balance.", "small error-text");
        return;
      }
      var b = body();
      if (b === lastBody) return;
      lastBody = b;
      if (ctrl) ctrl.abort();
      ctrl = typeof AbortController === "function" ? new AbortController() : null;
      var mine = ++seq;
      panel.hidden = false;
      panel.classList.add("tr-bal-busy");
      if (!panel.firstChild) say("Previewing the balance…");
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
        render(j);
      }).catch(function (e) {
        if (mine !== seq || (e && e.name === "AbortError")) return;
        panel.classList.remove("tr-bal-busy");
        lastBody = null;
        say("No preview: " + e.message, "small error-text");
      });
    };
    var later = function () {
      if (timer) clearTimeout(timer);
      timer = setTimeout(run, 400);
    };
    // Every field of the Datasets step (folders, scenes, classes, excluded, public, balance);
    // the class picker's changes come as tr-classes. An unchanged form asks nothing.
    var step = document.getElementById("datasets") || form;
    step.addEventListener("change", later);
    step.addEventListener("input", later);
    form.addEventListener("tr-classes", later);
    if (capOn) capOn.addEventListener("change", syncCap);
    syncCap();
    later();
  })();

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
  // retry that lowered it), input size, learning rate and schedule, epoch cap, patience, seed,
  // scene, data loader workers. From progress.settings (what the job resolved) and the trainer's
  // start event (batch_size, imgsz); a run queued before those existed falls back to its
  // request's recipe. Missing parts are left out; nothing known = no line.
  function lrText(v) { return v === 0 ? "0" : (+v.toPrecision(4)).toExponential(); }
  var isNum = function (v) { return typeof v === "number" && !isNaN(v); };
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
  function loadMetrics() {
    if (!document.querySelector(".tr-metrics[data-run]")) return;
    fetch("/v1/training", { headers: { Accept: "application/json" } })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) { if (data) { renderSettings(data); renderMetrics(data, false); } })
      .catch(function () {});
  }
  loadMetrics();

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
      // The actual batch / input size arrive with the trainer's start event: kept current.
      if (attr === "run") settingsLine(tr, r);
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
        finish.hidden = !((r.state === "running" || r.state === "paused") && r.stage === "train" && !p.stopped && !p.phase);
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
