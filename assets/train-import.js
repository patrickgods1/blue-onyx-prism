// The Import tab of the Train area (footage import): the form's source show / hide (without JS
// every field shows and works), the preview of what the path holds (GET
// /v1/training/import/preview, debounced, the latest answer wins), and the auto-label run an
// import chains: a ticked "Auto-label after import" posts the settings of the last Auto-label run
// (TrainArea.alOptions, else the form's defaults) as `al.<name>` hidden inputs, with the classes
// picked here; the box and its follow-up options are remembered in localStorage. The Jobs card
// is TrainArea.follow (train-area.js).
(function () {
  "use strict";
  var TA = TrainArea;
  // Runs first: the blocks below read the restored fields.
  TA.restoreForm("import-form");
  // Set by the auto-label classes block: the picker's selection, whether some are outside the
  // class list, and the confirm that appends them (null without the picker).
  var alClasses = null;

  TA.toggle(document.getElementById("import-form"), "source", "data-source");
  TA.openInvalid();

  // The import preview: what the path holds and the per-video budget, one line under the
  // field (GET /v1/training/import/preview), debounced; Browse fills the field and fires
  // `input`, so a picked path is previewed too.
  var importForm = document.getElementById("import-form");
  var preview = document.getElementById("import-preview");
  if (importForm && preview) {
    var pathField = importForm.elements.path;
    var timer = null;
    var loads = window.PrismPoll.latest();
    var show = function (text, warn) {
      preview.textContent = text;
      preview.hidden = !text;
      preview.classList.toggle("error-text", !!warn);
    };
    var run = function () {
      var p = pathField ? pathField.value.trim() : "";
      if (!p) { loads.stop(); show("", false); return; }
      var t = loads.begin();
      var q = "path=" + encodeURIComponent(p);
      var mc = importForm.elements.max_candidates;
      if (mc && mc.value.trim()) q += "&max_candidates=" + encodeURIComponent(mc.value.trim());
      var cam = importForm.elements.camera;
      if (cam && cam.value.trim()) q += "&camera=" + encodeURIComponent(cam.value.trim());
      show("Checking…", false);
      fetch("/v1/training/import/preview?" + q, Object.assign({ headers: { Accept: "application/json" } }, t.init))
        .then(TA.asJson)
        .then(function (j) {
          if (!t.current()) return;
          if (!j.success) { show(j.message || "Cannot read this path.", true); return; }
          var w = (j.warnings || []).filter(function (x) { return x.indexOf(".bvr") < 0; });
          show(j.summary + (w.length ? ". " + w.join(". ") : ""), false);
        })
        .catch(function (e) {
          if (t.current()) show("The path could not be checked (" + e.message + "). Import still works.", true);
        });
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

  // Auto-label classes on the Import tab: the class picker, for a run the import chains.
  (function () {
    var holder = document.getElementById("import-al-picker"), CP = window.ClassPicker;
    if (!holder || !CP) return;
    var picker = null, catalog = null;
    CP.fetchCatalog().then(function (cat) {
      catalog = cat;
      picker = CP.create(holder, {
        entries: TA.alEntries(cat), selected: CP.util.defaultSelection(cat), mode: "compact", label: "Classes", allowAdd: true,
        quickSets: CP.util.quickSetsFor(cat)
      });
    }).catch(function (e) {
      holder.textContent = "";
      var p = document.createElement("p");
      p.className = "muted small";
      p.textContent = "The class list could not be read (" + e.message + "); the labelers use the class list and training.synonyms.";
      holder.appendChild(p);
    });
    alClasses = function () { return picker ? TA.alSelection(picker, catalog, CP) : null; };
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
      TA.alOptions().forEach(function (p) { add(p[0], p[1]); });
      if (cls) cls.names.forEach(function (n) { add("classes", n); });
    });
  })();

  TA.follow();
})();
