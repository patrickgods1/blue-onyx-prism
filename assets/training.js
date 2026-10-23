// Training page: live progress of the queued and running training runs and import / auto-label
// jobs (polls GET /v1/training) and a reload when one finishes, so the results show.
(function () {
  "use strict";
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
