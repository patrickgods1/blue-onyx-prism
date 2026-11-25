// The Models tab of the Train area, Use card: "Use as default" draws the default switch's preview
// (what changes, the copied settings, the suggested threshold; GET /v1/training/switch-preview)
// in the row under the model and asks before a low-confidence threshold is saved, and "Use as
// verifier" asks before replacing the verifier it names. The Jobs card and the runs' metrics are
// TrainArea.follow (train-area.js).
(function () {
  "use strict";
  var TA = TrainArea;
  TA.openInvalid();

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
      // A low-confidence suggestion (s.confirm) is saved only with confirm=1: asked in the page
      // first, naming why, when the form would save it (always when already the default, else
      // with "Use the suggested threshold" ticked). Without the answer the server refuses.
      if (s && s.confirm) {
        var ask = mk("div", "warn"), yes = mk("button", null, "Save anyway"), no = mk("button", "secondary", "Cancel"), answered = false;
        ask.setAttribute("role", "alertdialog"); ask.hidden = true;
        yes.type = "button"; no.type = "button";
        var acts = mk("div", "actions"); acts.appendChild(yes); acts.appendChild(no);
        var why = String(s.low_confidence || "too little data to cross-validate").replace(/^low confidence:\s*/i, "").trim();
        ask.appendChild(mk("p", null, "The suggested threshold " + s.threshold.toFixed(2) + " is low confidence: " + why + (/[.!?]$/.test(why) ? "" : ".") + " Save it anyway?"));
        ask.appendChild(acts);
        form.insertBefore(ask, bar);
        form.addEventListener("submit", function (e) {
          if (answered || !(already || (use && use.input.checked))) return;
          e.preventDefault(); e.stopImmediatePropagation();
          ask.hidden = false; yes.focus();
        });
        yes.addEventListener("click", function () {
          answered = true; ask.hidden = true;
          form.appendChild(hidden("confirm", "1"));
          if (form.requestSubmit) form.requestSubmit(go); else form.submit();
        });
        no.addEventListener("click", function () { ask.hidden = true; go.focus(); });
        if (use) use.input.addEventListener("change", function () { if (!use.input.checked) ask.hidden = true; });
      }
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

  TA.follow();
})();
