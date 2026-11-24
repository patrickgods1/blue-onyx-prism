// Blue Onyx Prism: Home page (served from /static/; no inline scripts under the CSP).
// While a model loads or waits for a download, follow its state (and reload once it changes).
// Rows are matched by `data-model`, never by position: the table may be sorted (sort.js).
(function () {
  var rows = document.querySelectorAll("table.models tbody tr");
  var busy = Array.prototype.some.call(rows, function (r) { return r.className === "st-loading" || r.className === "st-wait"; });
  if (!busy) return;
  var poller = PrismPoll.create({period: 2000, run: function () {
    return fetch("/stats.json", {cache: "no-store"}).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (j) {
      var changed = false, byName = {};
      Array.prototype.forEach.call(rows, function (r) { byName[r.dataset.model] = r; });
      j.models.forEach(function (m) {
        var row = byName[m.name];
        if (!row) return;
        if (row.className !== "st-" + m.stateClass) changed = true;
        var d = row.querySelector(".detail");
        if (d && m.stateDetail) d.textContent = m.stateDetail;
      });
      // Reload where the user was (PrismLive keeps the scroll position across it).
      // A reload this poll makes, not the user: no activity (live.js).
      if (changed) { poller.stop(); if (window.PrismLive) PrismLive.reloadKeep(null, true); else PrismPoll.navigate(null); }
    });
  }});
  poller.start();
})();
// Pending restart badges follow the live config (another tab, Benchmark apply, the CLI).
if (window.PrismLive) PrismLive.onConfig(function (j) {
  var byKey = {}; j.models.forEach(function (m) { byKey[m.key] = m; });
  document.querySelectorAll("table.models .pending-badge[data-key]").forEach(function (b) {
    // `data-key` is the server's `normalize_name` of the model, as `m.key` is.
    var m = byKey[b.dataset.key];
    var reasons = m ? m.reasons : ["removed from the config; still loaded"];
    b.hidden = !reasons.length; b.dataset.tip = reasons.join("; ");
  });
});
fetch("/v1/status/updateavailable").then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }).then(u => {
  if (u.success && u.updateAvailable && u.latest) {
    const el = document.getElementById("update-hint");
    el.textContent = "Update available: " + u.latest.major + "." + u.latest.minor + "." + u.latest.patch;
    el.hidden = false;
  }
}).catch(() => {});
