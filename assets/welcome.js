// Blue Onyx Prism: Home page (served from /static/; no inline scripts under the CSP).
// While a model loads or waits for a download, follow its state (and reload once it changes).
(function () {
  var rows = document.querySelectorAll("table.models tbody tr");
  var busy = Array.prototype.some.call(rows, function (r) { return r.className === "st-loading" || r.className === "st-wait"; });
  if (!busy) return;
  var timer = setInterval(function () {
    if (document.hidden) return;
    fetch("/stats.json", {cache: "no-store"}).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (j) {
      var changed = false;
      j.models.forEach(function (m, i) {
        var row = rows[i];
        if (!row) return;
        if (row.className !== "st-" + m.stateClass) changed = true;
        var d = row.querySelector(".detail");
        if (d && m.stateDetail) d.textContent = m.stateDetail;
      });
      // Reload where the user was (PrismLive keeps the scroll position across it).
      if (changed) { clearInterval(timer); if (window.PrismLive) PrismLive.reloadKeep(); else location.reload(); }
    }).catch(function () {});
  }, 2000);
})();
// Pending restart badges follow the live config (another tab, Benchmark apply, the CLI).
if (window.PrismLive) PrismLive.onConfig(function (j) {
  var norm = function (n) { n = n.trim().toLowerCase(); var m = /^(.+)\.(onnx|xml)$/.exec(n); return m ? m[1] : n; };
  var byKey = {}; j.models.forEach(function (m) { byKey[m.key] = m; });
  document.querySelectorAll("table.models .pending-badge[data-model]").forEach(function (b) {
    var m = byKey[norm(b.dataset.model)];
    var reasons = m ? m.reasons : ["removed from the config; still loaded"];
    b.hidden = !reasons.length; b.title = reasons.join("; ");
  });
});
fetch("/v1/status/updateavailable").then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }).then(u => {
  if (u.success && u.updateAvailable && u.latest) {
    const el = document.getElementById("update-hint");
    el.textContent = "Update available: " + u.latest.major + "." + u.latest.minor + "." + u.latest.patch;
    el.hidden = false;
  }
}).catch(() => {});
