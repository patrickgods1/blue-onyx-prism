// Blue Onyx Prism: Stats page (served from /static/; no inline scripts under the CSP).
// Update the table in place from /stats.json (keeps scroll position, no flicker).
(function () {
  var body = document.getElementById("stats-body"), live = document.getElementById("live");
  function td(text, cls, title) {
    var c = document.createElement("td");
    c.textContent = text;
    if (cls) c.className = cls;
    if (title) c.title = title;
    return c;
  }
  function stat(s) {
    if (!s || !s.count) return ["-", "-", "-"];
    return [s.avg.toFixed(1), String(s.min), String(s.max)];
  }
  function update() {
    if (document.hidden) return;
    fetch("/stats.json", {cache: "no-store"}).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (j) {
      document.getElementById("uptime").textContent = j.uptime;
      var frag = document.createDocumentFragment();
      j.models.forEach(function (m) {
        var tr = document.createElement("tr");
        tr.className = "st-" + m.stateClass;
        var name = td(m.name);
        if (m.default) {
          var t = document.createElement("span");
          t.className = "tag"; t.textContent = "default";
          name.appendChild(document.createTextNode(" ")); name.appendChild(t);
        }
        tr.appendChild(name);
        // Short state, then the failure / download / compile detail on its own line (as Home).
        var st = td(m.stateShort || m.stateText, "state");
        if (m.stateShort && m.stateDetail) {
          var d = document.createElement("span");
          d.className = "detail"; d.textContent = m.stateDetail;
          st.appendChild(document.createElement("br")); st.appendChild(d);
        }
        tr.appendChild(st);
        tr.appendChild(td(m.runtime));
        tr.appendChild(td(m.executionProvider, "trunc", m.device || m.requestedDevice));
        tr.appendChild(td(m.fellBack === null || m.fellBack === undefined ? "-" : (m.fellBack ? "yes" : "no")));
        tr.appendChild(td(String(m.requests), "num"));
        tr.appendChild(td(String(m.dropped), "num"));
        tr.appendChild(td(m.queueLength + "/" + m.queueCapacity, "num"));
        [m.inferenceMs, m.processMs, m.roundTripMs].forEach(function (s) {
          stat(s).forEach(function (v) { tr.appendChild(td(v, "num")); });
        });
        frag.appendChild(tr);
      });
      // Same models in the same order: update the cells that changed in place (keeps a text
      // selection and focus elsewhere in the table); otherwise replace the rows.
      var fresh = Array.prototype.slice.call(frag.children), rows = body.rows;
      var same = rows.length === fresh.length && fresh.every(function (tr, i) { return rows[i].cells.length === tr.cells.length && rows[i].cells[0].textContent === tr.cells[0].textContent; });
      if (!same) body.replaceChildren(frag);
      else fresh.forEach(function (tr, i) {
        var old = rows[i];
        if (old.className !== tr.className) old.className = tr.className;
        for (var k = 0; k < tr.cells.length; k++) {
          var a = old.cells[k], b = tr.cells[k];
          if (a.innerHTML !== b.innerHTML) a.replaceWith(b.cloneNode(true));
          else if (a.title !== b.title) a.title = b.title;
        }
      });
      live.textContent = "updated " + new Date().toLocaleTimeString();
    }).catch(function (e) { live.textContent = "server not answering" + (e && /^HTTP /.test(e.message) ? " (" + e.message + ")" : ""); });
  }
  setInterval(update, 5000);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) update(); });
})();
