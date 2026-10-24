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
  // block's fields still post; the server ignores the ones the choice does not use.
  function toggle(form, name, attr) {
    if (!form) return;
    var update = function () {
      var v = checked(form, name);
      form.querySelectorAll("[" + attr + "]").forEach(function (el) {
        el.hidden = el.getAttribute(attr).split(" ").indexOf(v) < 0;
      });
    };
    form.querySelectorAll('input[name="' + name + '"]').forEach(function (r) {
      r.addEventListener("change", update);
    });
    update();
  }
  toggle(document.getElementById("import-form"), "source", "data-source");
  toggle(document.getElementById("autolabel-form"), "mode", "data-mode");

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
        if (a || b) window.location.reload();
      })
      .catch(function () {});
  }
  setInterval(tick, 3000);
})();
