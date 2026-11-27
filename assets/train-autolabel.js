// The Auto-label tab of the Train area: the form's mode show / hide (without JS every field shows
// and works), the VLM check's options, the Qwen3-VL sizes the GPU cannot hold, the SAM 3 threshold
// and the three-step SAM 3 access panel (GET /v1/autolabel/sam3; its option in the labeler select
// stays disabled until the checkpoint is installed), the class picker (classes outside the class
// list appended after a confirm) and the model picker of model mode.
// A link to /train/autolabel?<filters> (Review's "Auto-label these frames", a Recent imports row)
// fills the form's Which frames. The line above the button says what a run looks at, and the
// "Which frames" select under Advanced writes Status and Auto (the fields the form posts; the
// select itself has no name, so the import chain's `al.*` copy and the form's restore are the
// fields as before); Custom shows them. Preview frames posts the form to
// POST /training/autolabel/preview on a click (it reads every sidecar; the latest answer wins) and
// draws the answer. Each run's settings are kept (TrainArea.keepAlOptions) for the run an import
// chains on the Import tab. The Jobs card is TrainArea.follow (train-area.js).
(function () {
  "use strict";
  var TA = TrainArea;
  // Runs first: the blocks below read the restored fields.
  var restoredClasses = TA.restoreForm("autolabel-form");
  // Set by the auto-label classes block: the picker's selection, whether some are outside the
  // class list, and the confirm that appends them (null without the picker).
  var alClasses = null;

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
    ["bucket", "source", "import", "date_from", "date_to", "time_from", "time_to"].forEach(function (k) {
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

  TA.openInvalid();
  TA.toggle(document.getElementById("autolabel-form"), "mode", "data-mode");
  // Auto-label: the VLM check's options only while it is on (and the minimum score only while
  // ticked); the form's active mode for the CSS (labeler badges hide in model mode). Runs after
  // the mode toggle above, which re-enables every field of the shown mode.
  (function () {
    var form = document.getElementById("autolabel-form");
    if (!form) return;
    var vlm = form.elements.vlm, minOn = form.elements.vlm_min_on, min = form.elements.vlm_min_score;
    var opts = document.getElementById("al-vlm-options");
    var update = function () {
      form.dataset.active = TA.checked(form, "mode");
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
        o.textContent = over ? o.dataset.label + " – needs " + gb(need) : o.dataset.label;
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
      var on = TA.checked(form, "mode") === "labelers" && seat.value === "sam3";
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
    var sizeText = "";
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
    };
    var fail = function (text) {
      msg.textContent = text;
      msg.className = "form-msg error-text";
    };
    // One chained poll (live.js): it goes on after a failed answer (backing off), waits while the
    // tab is hidden, parks when nothing runs and does not keep the session alive. act() kicks it
    // after drawing a busy state.
    var poll = window.PrismPoll.create({
      period: 2000,
      run: function () {
        return fetch(API, { headers: { Accept: "application/json" } })
          .then(function (res) {
            if (!res.ok && res.status !== 409) throw new Error("HTTP " + res.status);
            return res.json();
          })
          .then(function (s) {
            if (!s || !s.state) throw new Error("no state");
            draw(s);
            return s;
          });
      },
      more: function (s) { return s.state === "checking" || s.state === "downloading"; },
      onError: function () { msg.textContent = "The server is not answering; trying again."; msg.className = "form-msg error-text"; }
    });
    // The answer of a refused action is a state too (409 needs_login, ...), so it is drawn the same.
    var act = function (path, button) {
      button.disabled = true;
      fetch(API + path, { method: "POST", headers: { Accept: "application/json" } })
        .then(TA.asJson)
        .then(function (s) {
          button.disabled = false;
          if (s.state) {
            draw(s);
            if (s.state === "checking" || s.state === "downloading") poll.kick();
          } else fail(s.message || "The server gave no answer.");
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
    poll.start();
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
        entries: TA.alEntries(cat), selected: restoredClasses || CP.util.defaultSelection(cat),
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
    alClasses = function () { return picker ? TA.alSelection(picker, catalog, CP) : null; };
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
      var skipped = !box || !box.checked ? [] : list.filter(function (c) { return names.indexOf(c) < 0 && !CP.util.isHidden(c, hidden); });
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

  // The settings of this form's run, for the run an import chains (TrainArea.alOptions).
  (function () {
    var form = document.getElementById("autolabel-form");
    if (form) TA.keepAlOptions(form);
  })();

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
  // which no longer describes it (and stops a preview still out). The classes are the picker's;
  // one not in the class list yet is appended only by a real run, so it is left out here and said
  // so.
  (function () {
    var form = document.getElementById("autolabel-form");
    var run = document.getElementById("al-preview-run");
    var box = document.getElementById("al-preview");
    if (!form || !run || !box) return;
    // "Start anyway" beside a memory refusal (PrismMemory, live.js): shown only with one.
    var anyway = document.getElementById("al-memory-anyway");
    var memory = function (m) { window.PrismMemory.sync(anyway, m); };
    var loads = window.PrismPoll.latest();
    var line = function (cls, text) {
      var p = document.createElement("p");
      p.className = cls;
      p.textContent = text;
      box.appendChild(p);
    };
    var clear = function () { loads.stop(); box.textContent = ""; box.hidden = true; memory(null); };
    var draw = function (j, note) {
      box.textContent = "";
      box.hidden = false;
      memory(j.memory);
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
    // Ticking "Start anyway" is no change to what the preview describes.
    form.addEventListener("change", function (e) {
      if (anyway && anyway.contains(e.target)) return;
      clear();
    });
    run.addEventListener("click", function () {
      var t = loads.begin();
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
      line("small muted", "Counting the frames…");
      run.disabled = true;
      fetch("/training/autolabel/preview", Object.assign({
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
        body: body.toString()
      }, t.init)).then(function (res) { return res.json(); }).then(function (j) {
        if (!t.current()) return;
        if (j.success) draw(j, note); else fail(j.message || "The server gave no answer.");
      }).catch(function (e) {
        if (t.current()) fail(e.message);
      }).then(function () { run.disabled = false; });
    });
  })();

  TA.follow();
})();
