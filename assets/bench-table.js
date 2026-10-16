// Blue Onyx Prism: Benchmark tables (PrismBench.dataTable, PrismBench.tableFilter).
// dataTable({key, columns, rows, filters, ...}) builds a sortable, filterable table: click a
// header to sort (aria-sort and an arrow; Enter / Space too), a text filter, model / device /
// runtime selects and "Only this run", a "Showing N of M" count and "Clear filters". Sort and
// filters live in per-table state (remembered per browser), so the 1 s polling re-render of a
// section rebuilds the table as the user left it. tableFilter(table, key) adds the text filter
// and the count to an existing table (the expert tables under "More detail").
// Loaded after bench-glossary.js (uses PB.el, PB.store, PB.head).
(function () {
  var PB = window.PrismBench, el = PB.el, has = PB.has;
  var states = {};
  function stateOf(key, dflt) {
    if (!states[key]) {
      var s = PB.store.get("table:" + key, null) || {};
      states[key] = {sort: s.sort || dflt || null, q: s.q || "", model: s.model || "", device: s.device || "", runtime: s.runtime || "", thisRun: !!s.thisRun};
    }
    return states[key];
  }
  function save(key) { PB.store.set("table:" + key, states[key]); }
  function cmp(a, b) {
    var na = a === null || a === undefined || (typeof a === "number" && isNaN(a)), nb = b === null || b === undefined || (typeof b === "number" && isNaN(b));
    if (na || nb) return na && nb ? 0 : na ? 1 : -1; // empty values last, either direction
    if (typeof a === "number" && typeof b === "number") return a - b;
    return String(a).localeCompare(String(b), undefined, {numeric: true});
  }
  function selectFilter(labelText, all, values, cur, onChange) {
    var l = el("label", "dt-filter"), s = el("select");
    l.appendChild(el("span", "visually-hidden", labelText));
    var o0 = el("option", null, all); o0.value = ""; s.appendChild(o0);
    values.forEach(function (v) { var o = el("option", null, v); o.value = v; if (v === cur) o.selected = true; s.appendChild(o); });
    if (cur && values.indexOf(cur) < 0) { var ox = el("option", null, cur + " (none shown)"); ox.value = cur; ox.selected = true; s.appendChild(ox); }
    s.addEventListener("change", function () { onChange(s.value); });
    l.appendChild(s); return l;
  }
  function distinct(rows, fn) { var seen = {}, out = []; rows.forEach(function (r) { var v = fn(r); if (has(v) && v !== "" && !seen[v]) { seen[v] = 1; out.push(String(v)); } }); return out.sort(function (a, b) { return a.localeCompare(b, undefined, {numeric: true}); }); }

  // o: key, cls, columns [{id, head ("Text" | "@labelId"), value(row) (sort key), render(row, td)
  // (returns text / node or fills td), num, label (mobile card label), first ("desc")}],
  // rows, filters {model(row), device(row), runtime(row), thisRun(row)} (each optional),
  // text(row) (what the text filter searches), sort {col, dir} (default), rowAttrs(row, tr),
  // empty (text when nothing matches), note.
  function dataTable(o) {
    var st = stateOf(o.key, o.sort), f = o.filters || {};
    var box = el("div", "dtable"); box.dataset.keep = "dt:" + o.key;
    var tools = el("div", "dt-tools"), count = el("span", "dt-count muted"), clear = el("button", "small secondary dt-clear", "Clear filters"); clear.type = "button";
    var q = el("input"); q.type = "search"; q.name = "dt-q-" + o.key; q.placeholder = "Filter…"; q.value = st.q; q.setAttribute("aria-label", "Filter the rows of this table");
    q.addEventListener("input", function () { st.q = q.value; save(o.key); draw(); });
    tools.appendChild(q);
    var selects = [];
    [["model", "Model", "All models"], ["device", "Device", "All devices"], ["runtime", "Runtime", "All runtimes"]].forEach(function (d) {
      if (!f[d[0]]) return;
      var vals = distinct(o.rows, f[d[0]]); if (vals.length < 2 && !st[d[0]]) return;
      var s = selectFilter(d[1], d[2], vals, st[d[0]], function (v) { st[d[0]] = v; save(o.key); draw(); });
      selects.push(s.querySelector("select")); tools.appendChild(s);
    });
    var tr0 = null;
    if (f.thisRun && o.rows.some(function (r) { return f.thisRun(r) === false; })) {
      var l = el("label", "check inline dt-filter"); tr0 = el("input"); tr0.type = "checkbox"; tr0.checked = st.thisRun;
      tr0.addEventListener("change", function () { st.thisRun = tr0.checked; save(o.key); draw(); });
      l.appendChild(tr0); l.appendChild(document.createTextNode(" Only this run")); tools.appendChild(l);
    }
    tools.appendChild(count); tools.appendChild(clear);
    clear.addEventListener("click", function () {
      st.q = ""; st.model = st.device = st.runtime = ""; st.thisRun = false; save(o.key);
      q.value = ""; selects.forEach(function (s) { s.value = ""; }); if (tr0) tr0.checked = false; draw();
    });
    box.appendChild(tools);
    if (o.note) box.appendChild(el("p", "muted dt-note", o.note));
    var t = el("table", "bench dt " + (o.cls || "")), th = el("thead"), hr = el("tr"), tb = el("tbody");
    o.columns.forEach(function (c) {
      var h = el("th", c.num ? "num" : null);
      if (typeof c.head === "string" && c.head.charAt(0) === "@") h.appendChild(PB.head(c.head.slice(1))); else if (c.head) h.appendChild(document.createTextNode(c.head));
      if (c.value) {
        h.classList.add("dt-sort"); h.tabIndex = 0;
        var go = function () {
          if (st.sort && st.sort.col === c.id) st.sort = {col: c.id, dir: st.sort.dir === "asc" ? "desc" : "asc"};
          else st.sort = {col: c.id, dir: c.first || "asc"};
          save(o.key); draw();
        };
        h.addEventListener("click", go);
        h.addEventListener("keydown", function (e) { if (e.target !== h) return; if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
        h.title = "Sort by " + (h.textContent || c.id);
      }
      h.dataset.col = c.id; hr.appendChild(h);
    });
    th.appendChild(hr); t.appendChild(th); t.appendChild(tb);
    var w = el("div", "scroll"); w.dataset.keep = "w:dt:" + o.key; w.appendChild(t); box.appendChild(w);
    var empty = el("p", "empty dt-empty"); box.appendChild(empty);
    function match(r) {
      if (st.model && f.model && String(f.model(r)) !== st.model) return false;
      if (st.device && f.device && String(f.device(r)) !== st.device) return false;
      if (st.runtime && f.runtime && String(f.runtime(r)) !== st.runtime) return false;
      if (st.thisRun && f.thisRun && f.thisRun(r) === false) return false;
      if (st.q) {
        var text = (o.text ? o.text(r) : [f.model && f.model(r), f.device && f.device(r), f.runtime && f.runtime(r)].join(" ")).toLowerCase();
        if (st.q.toLowerCase().split(/\s+/).some(function (w2) { return w2 && text.indexOf(w2) < 0; })) return false;
      }
      return true;
    }
    function draw() {
      var rows = o.rows.filter(match), col = st.sort && o.columns.find(function (c) { return c.id === st.sort.col && c.value; });
      if (col) {
        var dir = st.sort.dir === "desc" ? -1 : 1;
        rows = rows.map(function (r, i) { return {r: r, i: i, v: col.value(r)}; }).sort(function (a, b) {
          var na = a.v === null || a.v === undefined, nb = b.v === null || b.v === undefined;
          if (na || nb) return na && nb ? a.i - b.i : na ? 1 : -1;
          return dir * cmp(a.v, b.v) || a.i - b.i;
        }).map(function (x) { return x.r; });
      }
      hr.querySelectorAll("th").forEach(function (h) {
        var on = col && h.dataset.col === col.id;
        if (on) h.setAttribute("aria-sort", st.sort.dir === "desc" ? "descending" : "ascending"); else if (h.classList.contains("dt-sort")) h.setAttribute("aria-sort", "none");
      });
      tb.replaceChildren();
      rows.forEach(function (r) {
        var tr = el("tr");
        o.columns.forEach(function (c) {
          var td = el("td", c.num ? "num" : null); if (c.label || typeof c.head === "string") td.dataset.label = c.label || (c.head.charAt(0) === "@" ? PB.label(c.head.slice(1)) : c.head);
          var out = c.render ? c.render(r, td) : (c.value ? c.value(r) : "");
          if (out !== undefined && out !== null) { if (typeof out === "string" || typeof out === "number") td.appendChild(document.createTextNode(String(out))); else td.appendChild(out); }
          tr.appendChild(td);
        });
        if (o.rowAttrs) o.rowAttrs(r, tr);
        tb.appendChild(tr);
      });
      var active = !!(st.q || st.model || st.device || st.runtime || st.thisRun);
      count.textContent = "Showing " + rows.length + " of " + o.rows.length;
      clear.hidden = !active;
      empty.hidden = rows.length > 0; empty.textContent = o.rows.length ? "No rows match the filters; press Clear filters." : (o.empty || "Nothing to show yet.");
      w.hidden = !rows.length;
      if (o.after) o.after(rows);
    }
    draw();
    return box;
  }

  // A text filter and a "Showing N of M" count for an already built table (rows are hidden,
  // not removed, so its own sort keeps working). The filter text is remembered per key.
  function tableFilter(t, key) {
    var st = stateOf("tf:" + key), tools = el("div", "dt-tools"), q = el("input"), count = el("span", "dt-count muted");
    q.type = "search"; q.name = "dt-q-tf-" + key; q.placeholder = "Filter…"; q.value = st.q; q.setAttribute("aria-label", "Filter the rows of this table");
    function draw() {
      var rows = Array.prototype.slice.call(t.tBodies[0] ? t.tBodies[0].rows : []), words = st.q.toLowerCase().split(/\s+/).filter(Boolean), n = 0;
      rows.forEach(function (r) { var text = r.textContent.toLowerCase(), ok = words.every(function (w2) { return text.indexOf(w2) >= 0; }); r.hidden = !ok; if (ok) n++; });
      count.textContent = "Showing " + n + " of " + rows.length;
    }
    q.addEventListener("input", function () { st.q = q.value; save("tf:" + key); draw(); });
    tools.appendChild(q); tools.appendChild(count); draw();
    return tools;
  }

  PB.dataTable = dataTable; PB.tableFilter = tableFilter; PB.tableState = states;
})();
