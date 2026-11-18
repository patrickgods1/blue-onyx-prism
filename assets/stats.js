// Blue Onyx Prism: Stats page (served from /static/; no inline scripts under the CSP).
// Update the table in place from /stats.json (keeps scroll position, no flicker). Rows are
// matched by `data-model` (the table may be sorted, sort.js): the same models are patched where
// they are and the user's sort is applied again; only another set of models replaces the rows.
(function () {
  var body = document.getElementById("stats-body"), live = document.getElementById("live");
  var table = body && body.closest("table");
  // `label`: the column's name on a phone's card, the same as the template's data-label.
  function td(text, cls, title, label) {
    var c = document.createElement("td");
    c.textContent = text;
    if (cls) c.className = cls;
    if (title) c.title = title;
    if (label) c.setAttribute("data-label", label);
    return c;
  }
  // One timing in one cell: the average, min-max under it (as the template).
  function stat(s, label) {
    if (!s || !s.count) return td("-", "num", null, label);
    var c = td(s.avg.toFixed(1), "num", "min " + s.min + " \u00b7 max " + s.max, label);
    var sub = document.createElement("span");
    sub.className = "sub"; sub.textContent = s.min + "\u2013" + s.max;
    c.appendChild(sub);
    return c;
  }
  function update() {
    return fetch("/stats.json", {cache: "no-store"}).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (j) {
      document.getElementById("uptime").textContent = j.uptime;
      var frag = document.createDocumentFragment();
      j.models.forEach(function (m) {
        var tr = document.createElement("tr");
        tr.className = "st-" + m.stateClass;
        tr.setAttribute("data-model", m.name);
        var name = td(m.name, "rc-title");
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
        tr.appendChild(td(m.runtime, null, null, "Runtime"));
        // The provider, the device spec under it (as the template; visible, not only a title).
        var dev = td(m.executionProvider, "trunc", m.executionProvider, "Device"), spec = document.createElement("span");
        spec.className = "sub"; spec.textContent = m.device || m.requestedDevice || "";
        dev.appendChild(spec); tr.appendChild(dev);
        tr.appendChild(td(m.fellBack === null || m.fellBack === undefined ? "-" : (m.fellBack ? "yes" : "no"), null, null, "Fell back"));
        tr.appendChild(td(String(m.requests), "num", null, "Requests"));
        tr.appendChild(td(String(m.dropped), "num", null, "Dropped"));
        tr.appendChild(td(m.queueLength + "/" + m.queueCapacity, "num", null, "Queue"));
        [[m.inferenceMs, "Inference ms"], [m.processMs, "Process ms"], [m.roundTripMs, "Round trip ms"]].forEach(function (s) { tr.appendChild(stat(s[0], s[1])); });
        frag.appendChild(tr);
      });
      // The same models (in any order): update the cells that changed in place (keeps a text
      // selection, focus elsewhere in the table and the user's sort); otherwise replace the rows.
      var fresh = Array.prototype.slice.call(frag.children), byName = {};
      Array.prototype.forEach.call(body.rows, function (r) { byName[r.getAttribute("data-model")] = r; });
      var same = body.rows.length === fresh.length && fresh.every(function (tr) {
        var old = byName[tr.getAttribute("data-model")];
        return !!old && old.cells.length === tr.cells.length;
      });
      if (!same) body.replaceChildren(frag);
      else fresh.forEach(function (tr) {
        var old = byName[tr.getAttribute("data-model")];
        if (old.className !== tr.className) old.className = tr.className;
        for (var k = 0; k < tr.cells.length; k++) {
          var a = old.cells[k], b = tr.cells[k];
          if (a.innerHTML !== b.innerHTML) a.replaceWith(b.cloneNode(true));
          else if (a.title !== b.title) a.title = b.title;
        }
      });
      if (window.PrismSort && table) PrismSort.apply(table);
      live.textContent = "updated " + PrismTime.clock(Date.now());
    }).catch(function (e) {
      live.textContent = "server not answering" + (e && /^HTTP /.test(e.message) ? " (" + e.message + ")" : "");
      throw e; // the poller backs off
    });
  }
  PrismPoll.create({run: update, period: 5000}).start();
})();
