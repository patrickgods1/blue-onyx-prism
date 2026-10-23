// Training page: live progress of the queued and running runs (polls GET /v1/training) and a
// reload when a run finishes, so the results table shows it.
(function () {
  "use strict";
  var table = document.getElementById("training-runs");
  if (!table) return;
  var busy = {};
  table.querySelectorAll("tr[data-run]").forEach(function (tr) {
    var st = tr.querySelector('[data-field="state"]');
    if (st && (st.textContent === "queued" || st.textContent === "running")) busy[tr.dataset.run] = true;
  });
  if (Object.keys(busy).length === 0) return;
  function describe(r) {
    if (r.state === "queued") return "queued";
    if (r.state !== "running") return r.state;
    var s = r.stage ? r.stage.replace(/_/g, " ") : "starting";
    return s + " (" + r.step + "/" + r.steps + ")" + (r.detail ? ": " + r.detail : "");
  }
  function tick() {
    fetch("/v1/training", { headers: { Accept: "application/json" } })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data) return;
        var finished = false;
        (data.runs || []).forEach(function (r) {
          var tr = table.querySelector('tr[data-run="' + r.id + '"]');
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
        if (finished) window.location.reload();
      })
      .catch(function () {});
  }
  setInterval(tick, 3000);
})();
