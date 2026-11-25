// The Train tab of the Train area: the runs line (one run per ticked scene per model, named
// before queuing), the Learning rate field's placeholder that follows the ticked family, the night
// augmentation fields, the class picker of the ticked folders (counts per scene, warnings, class
// merges; GET /v1/training/dataset-classes, the latest answer wins), the balance preview (what the
// Balance fieldset does to each run's training frames) and the pre-flight report above Queue
// training. A link to /train/train?dataset=<folder> ticks that export. The Jobs card (Finish now
// while training, the Stop now / Continue notice of an early-stop point passed while validation
// still improves, each run's settings line) is TrainArea.follow (train-area.js).
(function () {
  "use strict";
  var TA = TrainArea;
  // Runs first: the blocks below read the restored fields.
  TA.restoreForm("train");
  // Set by the train classes block: the picked classes in head order (null without the picker).
  var trainClasses = null;
  // Set by the train classes block: the active class merges as "from=to" strings (null without
  // the merge rows: the text field then posts them itself).
  var trainRename = null;

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

  TA.openInvalid();

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
    var RKEY = "bop.training.rename";
    var NEW = "__new__";
    var FEW = 50;
    var picker = null, inv = null, byName = {}, catalog = null, timer = null, lastQuery = null;
    var loads = window.PrismPoll.latest();
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
    // Merge classes: rows {from, to, kind} (kind class | drop | new), kept in localStorage; the
    // exports' own class names (with boxes) come with each answer as `source_classes`.
    var mergeBox = document.getElementById("train-merges");
    var rows = [], srcList = [], rowsEl = null;
    var storedRows = function () {
      try {
        var v = JSON.parse(window.localStorage.getItem(RKEY) || "null");
        if (!Array.isArray(v)) return [];
        return v.filter(function (r) { return r && typeof r.from === "string" && typeof r.to === "string"; })
          .map(function (r) { return { from: r.from, to: r.to, kind: r.kind === "drop" || r.kind === "new" ? r.kind : "class" }; });
      } catch (e) { return []; }
    };
    var storeRows = function () {
      try { window.localStorage.setItem(RKEY, JSON.stringify(rows)); } catch (e) { /* storage blocked */ }
    };
    var srcBoxes = function (n) {
      for (var i = 0; i < srcList.length; i++) if (srcList[i].name === n) return srcList[i].boxes;
      return null;
    };
    // A row that merges: from chosen (and in the exports, once they are known) and a target.
    var complete = function (r) {
      if (!r.from) return false;
      if (srcList.length && srcBoxes(r.from) === null) return false;
      if (r.kind === "drop") return true;
      return r.to.trim() !== "" && r.to.trim() !== r.from;
    };
    var activeRows = function () { return rows.filter(complete); };
    var activePairs = function () {
      var seen = {}, out = [];
      activeRows().forEach(function (r) {
        var to = r.kind === "drop" ? "" : r.to.trim();
        seen[r.from] = to;
      });
      Object.keys(seen).forEach(function (k) { out.push(k + "=" + seen[k]); });
      return out;
    };
    var renameQuery = function () {
      return activePairs().map(function (r) { return "rename=" + encodeURIComponent(r); });
    };
    if (mergeBox) trainRename = activePairs;
    // The merged-away classes that a class absorbed, for its chip.
    var absorbed = function (name) {
      var out = [];
      activeRows().forEach(function (r) {
        if (r.kind !== "drop" && r.to.trim() === name && srcBoxes(r.from)) out.push(srcBoxes(r.from) + " " + r.from);
      });
      return out;
    };
    var option = function (value, text, sel) {
      var o = mk("option", null, text);
      o.value = value;
      if (sel) o.selected = true;
      return o;
    };
    var renderRows = function () {
      if (!rowsEl) return;
      rowsEl.replaceChildren();
      rows.forEach(function (r, i) {
        var row = mk("div", "tr-merge-row");
        var gone = srcList.length && r.from && srcBoxes(r.from) === null;
        var del = mk("button", "tr-merge-x", "\u2715");
        del.type = "button";
        del.title = "Remove this merge";
        del.setAttribute("aria-label", "Remove this merge");
        del.addEventListener("click", function (ev) {
          ev.stopPropagation();
          rows.splice(i, 1);
          storeRows(); renderRows(); load();
        });
        if (gone) {
          row.classList.add("tr-merge-off");
          row.appendChild(mk("span", "small", r.from + " \u2192 " + (r.kind === "drop" ? "(drop)" : r.to)));
          row.appendChild(mk("span", "muted small", "no " + r.from + " boxes in these exports"));
          row.appendChild(del);
          rowsEl.appendChild(row);
          return;
        }
        var others = rows.filter(function (o, j) { return j !== i; });
        var otherFrom = others.map(function (o) { return o.from; }).filter(Boolean);
        var otherTo = others.filter(function (o) { return o.kind !== "drop" && o.to.trim(); }).map(function (o) { return o.to.trim(); });
        var from = mk("select", "tr-merge-from");
        from.setAttribute("aria-label", "Merge this class");
        from.appendChild(option("", "choose a class\u2026", !r.from));
        if (r.from && !srcList.length) from.appendChild(option(r.from, r.from, true));
        srcList.forEach(function (c) {
          if (c.name !== r.from && (otherFrom.indexOf(c.name) >= 0 || otherTo.indexOf(c.name) >= 0)) return;
          from.appendChild(option(c.name, c.name + " (" + c.boxes + ")", c.name === r.from));
        });
        var to = mk("select", "tr-merge-to");
        to.setAttribute("aria-label", "into this class");
        var tos = srcList.filter(function (c) {
          return c.name !== r.from && otherFrom.indexOf(c.name) < 0;
        });
        var cur = r.kind === "class" ? r.to : null;
        if (cur && !tos.some(function (c) { return c.name === cur; })) tos.push({ name: cur, boxes: 0 });
        to.appendChild(option("", "choose\u2026", r.kind === "class" && !r.to));
        tos.forEach(function (c) { to.appendChild(option(c.name, c.name, r.kind === "class" && c.name === r.to)); });
        to.appendChild(option("", "drop (remove these boxes)", r.kind === "drop"));
        to.lastChild.value = "__drop__";
        to.appendChild(option(NEW, "new name\u2026", r.kind === "new"));
        var name = mk("input", "tr-merge-new");
        name.type = "text"; name.autocapitalize = "none"; name.spellcheck = false;
        name.setAttribute("aria-label", "New class name");
        name.placeholder = "new class name";
        name.value = r.kind === "new" ? r.to : "";
        name.hidden = r.kind !== "new";
        var stopForm = function (ev) { ev.stopPropagation(); };
        row.addEventListener("input", stopForm);
        row.addEventListener("change", stopForm);
        from.addEventListener("change", function () {
          r.from = from.value;
          storeRows(); renderRows(); load();
        });
        to.addEventListener("change", function () {
          if (to.value === "__drop__") { r.kind = "drop"; r.to = ""; }
          else if (to.value === NEW) { r.kind = "new"; r.to = ""; }
          else { r.kind = "class"; r.to = to.value; }
          storeRows(); renderRows(); load();
        });
        name.addEventListener("input", function () { r.to = name.value; storeRows(); later(); });
        name.addEventListener("change", function () { load(); });
        row.appendChild(from);
        row.appendChild(mk("span", "tr-merge-arrow", "\u2192"));
        row.appendChild(to);
        row.appendChild(name);
        row.appendChild(del);
        rowsEl.appendChild(row);
      });
    };
    if (mergeBox) {
      var fallback = mergeBox.querySelector('input[name="rename"]');
      rowsEl = mk("div", "tr-merge-rows");
      var add = mk("button", "tr-merge-add", "Add a merge");
      add.type = "button";
      add.addEventListener("click", function () {
        rows.push({ from: "", to: "", kind: "class" });
        storeRows(); renderRows();
      });
      if (fallback) {
        fallback.parentNode.insertBefore(rowsEl, fallback);
        fallback.parentNode.insertBefore(add, fallback);
        fallback.remove();
      } else {
        mergeBox.appendChild(rowsEl);
        mergeBox.appendChild(add);
      }
      rows = storedRows();
      renderRows();
    }
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
      var inc = absorbed(c.name);
      if (inc.length) {
        var incl = mk("span", "cp-badge tr-cls-merged", "includes " + inc.join(", "));
        incl.title = "Boxes merged into " + c.name + " (Merge classes).";
        label.appendChild(incl);
      }
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
      var q = list.map(function (d) { return "dataset=" + encodeURIComponent(d); }).concat(renameQuery()).join("&");
      if (q === lastQuery) return;
      lastQuery = q;
      var t = loads.begin();
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
        fetch("/v1/training/dataset-classes?" + q, Object.assign({ headers: { Accept: "application/json" } }, t.init))
          .then(TA.asJson)
      ]).then(function (r) {
        if (!t.current()) return;
        catalog = r[0] || {};
        if (!r[1] || r[1].success === false) throw new Error((r[1] && r[1].message) || "no answer");
        inv = r[1];
        if (Array.isArray(inv.source_classes)) { srcList = inv.source_classes; renderRows(); }
        build();
      }).catch(function (e) {
        if (!t.current()) return;
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
      if (mergeBox && rowsEl) {
        activePairs().forEach(function (r) {
          var i = document.createElement("input");
          i.type = "hidden"; i.name = "rename"; i.value = r; i.className = "tr-cls-field";
          form.appendChild(i);
        });
      }
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
    var timer = null, lastBody = null, tab = 0;
    var loads = window.PrismPoll.latest();
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
      var rn = trainRename ? trainRename() : null;
      if (rn) rn.forEach(function (r) { p.append("rename", r); });
      else {
        var rt = form.querySelector('#train-merges input[name="rename"]');
        if (rt && rt.value.trim()) p.append("rename", rt.value.trim());
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
      // The disk estimate of the ticked models, and the data loader workers each would get
      // (learned per batch and input size, so those go too).
      form.querySelectorAll('input[name="model"]:checked').forEach(function (b) { p.append("model", b.value); });
      var seeds = form.elements.seeds, dir = form.elements.install_dir;
      if (seeds && seeds.value) p.append("seeds", seeds.value);
      if (dir && dir.value.trim()) p.append("install_dir", dir.value.trim());
      ["batch", "imgsz"].forEach(function (n) {
        var e = form.elements[n];
        if (e && e.value.trim()) p.append(n, e.value.trim());
      });
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
      // Sortable (sort.js), one remembered sort per kind of table ("Boxes per class", ...).
      if (window.PrismSort) window.PrismSort.attach(t, { key: "balance:" + head });
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
        if (window.PrismSort) window.PrismSort.attach(t, { key: "preflight-classes" });
      }
      // Frames with ignore regions (unsure, too small, under the size floor), by the run's mode
      // (`training.ignore`, set on the Refine tab): background keeps a frame with a label left and
      // drops the rest, drop-frame drops every one. Same split as the CLI's text (preflight.rs).
      var background = p.ignore_mode !== "drop-frame";
      var left = [
        [background ? 0 : p.ignore_frames, "with unsure or too-small boxes (mode: leave the frame out; change it on the Refine tab)"],
        [background ? p.ignore_only : 0, "with nothing but unsure or too-small objects (no label to train)"],
        [background ? p.ignore_valid_day : 0, "with unsure or too-small objects on camera days used for validation"],
        [p.parked_capped, "over the parked-spot cap"],
        [p.static_capped, "showing nothing but parked objects, over the cap per static spot (" + p.static_spots + " spot" + (p.static_spots === 1 ? "" : "s") + " in the data; change the cap on the Refine tab)"],
        [p.duplicates_dropped, "that repeat a frame of another export"],
        [p.scene_filtered, "outside this run's scenes"]
      ].filter(function (x) { return x[0] > 0; }).map(function (x) { return x[0] + " " + x[1]; });
      if (left.length) d.appendChild(mk("p", "muted small", "Left out of training: " + left.join("; ") + "."));
      if (p.ignore_kept > 0) {
        var per = Object.keys(p.ignore_boxes_background || {}).sort().map(function (k) { return k + " " + p.ignore_boxes_background[k]; });
        d.appendChild(mk("p", "muted small", "Trained with an unsure or too-small object left unlabelled (as background): " + p.ignore_kept + " frame" + (p.ignore_kept === 1 ? "" : "s") + (per.length ? " (" + per.join(", ") + ")" : "") + "."));
      }
      var back = Object.keys(p.floor_restored || {}).sort();
      if (back.length) {
        d.appendChild(mk("p", "muted small", "Under the export's size floor but not at this run's input, so trained as labels: " +
          back.map(function (k) { return k + " " + p.floor_restored[k]; }).join(", ") + "."));
      }
      if (p.hard_negatives > 0) d.appendChild(mk("p", "muted small", p.hard_negatives + " of the frames without objects were confirmed empty or had a box rejected on the Review tab."));
      // The static cap is said once, in the list above: its note (the same count, for the CLI's
      // text) names training.static_cap, which no other note does.
      var notes = (p.notes || []).filter(function (n) { return !(p.static_capped > 0 && n.indexOf("training.static_cap") >= 0); });
      if (notes.length) d.appendChild(list(notes, "muted small"));
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
      // The train log's own line (learned on this machine, never above what memory allows now).
      (j.workers || []).forEach(function (l) { pre.appendChild(mk("p", "muted small", l)); });
      // The memory commit check each run meets at its start, with "Start anyway" beside it.
      if (j.memory && j.memory.refusal) pre.appendChild(mk("p", "error-text small", "A run would be refused if it started now: " + j.memory.refusal));
      window.PrismMemory.sync(document.getElementById("train-memory-anyway"), j.memory);
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
        loads.stop();
        lastBody = null;
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
      var t = loads.begin();
      if (ax.length) {
        panel.hidden = false;
        panel.classList.add("tr-bal-busy");
        if (!panel.firstChild) say("Previewing the balance…");
      }
      fetch("/v1/training/balance-preview", Object.assign({
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
        body: b
      }, t.init)).then(function (res) {
        return res.json().catch(function () { return null; }).then(function (j) {
          if (!res.ok || !j || j.success === false) throw new Error((j && j.message) || "the server answered HTTP " + res.status);
          return j;
        });
      }).then(function (j) {
        if (!t.current()) return;
        panel.classList.remove("tr-bal-busy");
        renderPre(j);
        if (axes().length) render(j);
      }).catch(function (e) {
        // A newer preview (or a stop) took its place: aborted, or answered too late.
        if (!t.current()) return;
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

  TA.follow();
})();
