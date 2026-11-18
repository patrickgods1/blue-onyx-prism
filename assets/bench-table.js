// Blue Onyx Prism: Benchmark tables (PrismBench.dataTable, PrismBench.tableFilter).
// dataTable({key, columns, rows, filters, ...}) builds a sortable, filterable table: the sort is
// PrismSort's (sort.js: click a header, Shift-click for a second key, the phones' Sort by; each
// cell's data-v is its column's value(row)), plus a text filter, model / device / runtime selects
// and "Only this run", a "Showing N of M" count and "Clear filters". Filters live in per-table
// state and the sort in PrismSort's (both remembered per browser; an old saved sort here moved to
// PrismSort once), so the 1 s polling re-render of a section rebuilds the table as the user left
// it. tableFilter(table, key) adds the text filter and the count to an existing table (the expert
// tables under "More detail").
// Loaded after bench-glossary.js (uses PB.el, PB.store, PB.head).
(function () {
  var PB = window.PrismBench, el = PB.el, has = PB.has;
  var states = {};
  function stateOf(key) {
    if (!states[key]) {
      var s = PB.store.get("table:" + key, null) || {};
      states[key] = {q: s.q || "", model: s.model || "", device: s.device || "", runtime: s.runtime || "", thisRun: !!s.thisRun};
    }
    return states[key];
  }
  function save(key) { PB.store.set("table:" + key, states[key]); }
  // A column value as data-v (null and undefined are empty).
  function dataV(v) { return v === null || v === undefined || (typeof v === "number" && !isFinite(v)) ? "" : String(v); }
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
  // text(row) (what the text filter searches), sort {col, dir} (the default order; none = the
  // rows' own order), rowAttrs(row, tr),
  // empty (text when nothing matches), note.
  function dataTable(o) {
    var st = stateOf(o.key), f = o.filters || {};
    var box = el("div", "dtable"); box.dataset.keep = "dt:" + o.key;
    var tools = el("div", "dt-tools"), count = el("span", "dt-count muted"), clear = el("button", "small secondary dt-clear", "Clear filters"); clear.type = "button";
    var q = el("input"); q.type = "search"; q.setAttribute("enterkeyhint", "search"); q.setAttribute("autocapitalize", "none"); q.setAttribute("autocorrect", "off"); q.spellcheck = false; q.name = "dt-q-" + o.key; q.placeholder = "Filter…"; q.value = st.q; q.setAttribute("aria-label", "Filter the rows of this table");
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
      h.dataset.col = c.id; h.dataset.sortCol = c.id;
      if (!c.value) h.setAttribute("data-sort-off", "");
      else if (c.first) h.dataset.sortFirst = c.first;
      hr.appendChild(h);
    });
    th.appendChild(hr); t.appendChild(th); t.appendChild(tb);
    if (window.PrismSort) PrismSort.attach(t, {key: o.key, defaultKeys: o.sort ? [o.sort] : []});
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
      var rows = o.rows.filter(match);
      tb.replaceChildren();
      rows.forEach(function (r) {
        var tr = el("tr");
        o.columns.forEach(function (c) {
          var td = el("td", c.num ? "num" : null); if (c.label || typeof c.head === "string") td.dataset.label = c.label || (c.head.charAt(0) === "@" ? PB.label(c.head.slice(1)) : c.head);
          if (c.value) td.dataset.v = dataV(c.value(r));
          var out = c.render ? c.render(r, td) : (c.value ? c.value(r) : "");
          if (out !== undefined && out !== null) { if (typeof out === "string" || typeof out === "number") td.appendChild(document.createTextNode(String(out))); else td.appendChild(out); }
          tr.appendChild(td);
        });
        if (o.rowAttrs) o.rowAttrs(r, tr);
        tb.appendChild(tr);
      });
      if (window.PrismSort) PrismSort.apply(t);
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
    q.type = "search"; q.setAttribute("enterkeyhint", "search"); q.setAttribute("autocapitalize", "none"); q.setAttribute("autocorrect", "off"); q.spellcheck = false; q.name = "dt-q-tf-" + key; q.placeholder = "Filter…"; q.value = st.q; q.setAttribute("aria-label", "Filter the rows of this table");
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
