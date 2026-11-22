// Blue Onyx Prism: PrismSort, the one table sorter of every page (served from /static/, loaded by
// base.html after live.js; no inline scripts under the CSP; ES5, no build step).
// Click a column header to sort by it (again: the other direction); Shift-click adds the column
// as the next key (or flips it when it is one already), with rank numbers in the headers. Phones
// (max-width: 700px, where `table.rcards` and `table.bench.stack` hide their headers) get a
// "Sort by" select above the table. "Default order" (or "Best first" when the table's default
// starts with `best`) goes back to the table's own order.
// Two parts, like live.js: a pure first part (no DOM; tested in node by tests/it/js_sort.rs) and
// the page part below `if (typeof document === "undefined") return;`.
//
// Pure part:
// - value(cell): the sort key of a cell: `data-v` first (a number when numeric, else a lowercased
//   string), then a `<time datetime>` child (epoch ms), then the text: numbers with thousands
//   separators and a sign, `%`, the units ms / s (x 1000) / B / kB / MB / GB (decimal) / px / /s,
//   `a/b` (no spaces) as the ratio a/b, and an empty cell, `-`, `–`, `—` or `n/a` as empty; else
//   the lowercased text. Empty is `null`. A plain object {v?, datetime?, text} stands for a cell.
// - textOf(node): a cell's text for sorting: its first line (up to a `<br>` after some text),
//   without hidden elements, controls, `<details>`, badges (`.tag`), `.sub` / `.detail` lines and
//   action buttons (a `button.term` holds data and counts). Works on DOM nodes and node-likes.
// - order(items, keys, get): a stable multi-key sort. keys [{col, dir: "asc"|"desc"}]; `get(item,
//   col)` returns the item's sort value (default: item.keys[col], item.best for `best`). Numbers
//   come before strings and empties last, in both directions; the item's place in `items` is the
//   last tie-break, so `items` goes in in the table's own order. The pseudo-column `best` sorts by
//   the row's `data-best` ordinal (ascending whatever its dir; rows without one after, in order).
// - arrange(items, keys, get, role): order() with row groups. `role(item)` (default item.role):
//   "follow" keeps a row right behind the row before it (detail rows), "top" / "bottom" pins a
//   group to the start / end in its own order (zero-state and total rows), "last" sorts a group
//   among the other "last" ones after every ordinary group (Compare's models that trained on the
//   images). "group" leads a section: the rows after it, up to the next "group" row, are arranged
//   among themselves (with the roles above), each section keeps its place and its lead row stays
//   at its top (the Models tab's held-out groups: scores compare only inside one). Rows before the
//   first "group" row are a section of their own.
// - parseState(json): stored state, the current `[{col, dir}, ...]` or the Benchmark page's old
//   `{col, dir}`, as a clean
//   list; null when there is nothing to use. serialize(keys) is its inverse.
// - click(keys, col, shift, first): the keys after a click on `col` (Shift held or not; `first` is
//   the column's first direction). describe(keys, label): "Sorted by X, descending, then by Y,
//   ascending".
//
// Page part: PrismSort.attach(table, {key, defaultKeys, columns}) wires a table (once; again only
// re-applies). Server markup opts in with `data-sort="<key>"` on the <table> (attached when the
// page has loaded, and after PrismLive.view.restore swaps a section in); optional
// `data-sort-default="col:dir,..."` (or `best`), `data-sort-cols="name,name,-,..."` naming the leaf
// columns in order (`-` = not sortable; an empty entry keeps the index), or per leaf <th>
// `data-sort-col`, `data-sort-off`, `data-sort-first="desc"`, `data-sort-label`. Columns without a
// name are named by their leaf index; a column whose header has no text is not sortable. Headers
// of several rows (colSpan / rowSpan) work: a leaf is the header over one column in the last row.
// Rows: `data-sort-follow` (stays behind the row before), `data-sort-pin="top|bottom"`,
// `tr.lineup-out` (last), `data-sort-group` (a section's lead row: the rows up to the next one sort
// among themselves, sections keep their order), `data-best` (ordinal of the `best` pseudo-column;
// filled by Rust, never by a JS ranking; so no real column may be named `best`). JS adds a `button.th-sort` (arrow and rank) at the end of each sortable <th>, so a
// `button.term` there keeps its tooltip (PrismTip stops its clicks in the capture phase); `aria-sort`
// on the first key's <th>; one polite status line announces each change. State per browser per
// table: localStorage `prism.sort:<pathname>:<key>` (every access in try/catch: without storage the
// choice lasts for the page).
// PrismSort.apply(table) re-applies it after a re-render (rows a render added get the next places of
// the table's own order); PrismSort.state / restore carry it through PrismLive.view.capture /
// restore.
(function (root) {
  "use strict";
  var EMPTY = {"": 1, "-": 1, "–": 1, "—": 1, "n/a": 1};
  var NUM = "[-+−]?(?:(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?|\\.\\d+)(?:e[-+]?\\d+)?";
  var UNITS = {"": 1, "%": 1, ms: 1, s: 1000, px: 1, b: 1, kb: 1e3, mb: 1e6, gb: 1e9, "/s": 1};
  var NUMBER = new RegExp("^(" + NUM + ")\\s*(%|ms|s|px|b|kb|mb|gb|/s)?$", "i");
  var RATIO = new RegExp("^(" + NUM + ")/(" + NUM + ")$");
  var MAX_KEYS = 8;
  function toNumber(s) { return parseFloat(String(s).replace(/,/g, "").replace(/−/g, "-")); }
  function lower(s) { return String(s).toLowerCase(); }
  // A trimmed text as a sort value (null = empty).
  function parse(text) {
    var t = String(text === null || text === undefined ? "" : text).replace(/\s+/g, " ").trim(), low = t.toLowerCase();
    if (EMPTY[low]) return null;
    var m = NUMBER.exec(t);
    if (m) { var n = toNumber(m[1]) * UNITS[lower(m[2] || "")]; return isFinite(n) ? n : low; }
    var r = RATIO.exec(t);
    if (r) { var a = toNumber(r[1]), b = toNumber(r[2]); return b !== 0 && isFinite(a / b) ? a / b : null; }
    return low;
  }
  // A `data-v` value: a number when it is one, else a lowercased string (empty-like = null).
  function fromData(v) {
    if (typeof v === "number") return isFinite(v) ? v : null;
    var s = String(v).trim();
    if (EMPTY[s.toLowerCase()]) return null;
    if (/^[-+]?(\d|\.\d)/.test(s) && isFinite(Number(s))) return Number(s);
    return s.toLowerCase();
  }
  function value(c) {
    if (c === null || c === undefined) return null;
    if (c.v !== undefined && c.v !== null) return fromData(c.v);
    if (c.datetime) { var t = Date.parse(c.datetime); return isNaN(t) ? lower(c.datetime) : t; }
    return parse(c.text);
  }
  // ---- a cell's text -------------------------------------------------------------------------
  var SKIP_TAGS = {SELECT: 1, OPTION: 1, TEXTAREA: 1, INPUT: 1, SCRIPT: 1, STYLE: 1, TEMPLATE: 1, DETAILS: 1, PROGRESS: 1, SVG: 1, CANVAS: 1, IMG: 1};
  var SKIP_CLASS = /(^|\s)(sub|detail|tag|sr-only|visually-hidden|th-sort)(\s|$)/;
  function classOf(n) {
    if (typeof n.className === "string") return n.className;
    return (n.getAttribute && n.getAttribute("class")) || "";
  }
  function textOf(node) {
    var out = [], stop = false;
    function walk(n) {
      var kids = n.childNodes || [];
      for (var i = 0; i < kids.length && !stop; i++) {
        var c = kids[i];
        if (c.nodeType === 3) { out.push(c.nodeValue || ""); continue; }
        if (c.nodeType !== 1) continue;
        var tag = String(c.nodeName || "").toUpperCase();
        if (tag === "BR") { if (out.join("").trim()) stop = true; continue; }
        if (SKIP_TAGS[tag] || c.hidden) continue;
        var cls = classOf(c);
        if (SKIP_CLASS.test(cls)) continue;
        if (tag === "BUTTON" && !/(^|\s)term(\s|$)/.test(cls)) continue;
        walk(c);
      }
    }
    if (node) walk(node);
    return out.join("").replace(/\s+/g, " ").trim();
  }
  // ---- ordering ------------------------------------------------------------------------------
  var collator = null;
  try { collator = typeof Intl !== "undefined" && Intl.Collator ? new Intl.Collator(undefined, {numeric: true, sensitivity: "base"}) : null; } catch (e) { collator = null; }
  function collate(a, b) { return collator ? collator.compare(a, b) : a < b ? -1 : a > b ? 1 : 0; }
  function isEmpty(x) { return x === null || x === undefined || (typeof x === "number" && isNaN(x)); }
  // Empties last and numbers before strings whatever the direction; `dir` turns the rest.
  function compare(x, y, dir) {
    var ex = isEmpty(x), ey = isEmpty(y);
    if (ex || ey) return ex === ey ? 0 : ex ? 1 : -1;
    var nx = typeof x === "number", ny = typeof y === "number";
    if (nx !== ny) return nx ? -1 : 1;
    var c = nx ? (x < y ? -1 : x > y ? 1 : 0) : collate(String(x), String(y));
    return dir === "desc" ? -c : c;
  }
  function clean(list) {
    var out = [], seen = {};
    (Array.isArray(list) ? list : []).forEach(function (e) {
      if (!e || typeof e !== "object" || out.length >= MAX_KEYS) return;
      var col = e.col;
      if (typeof col === "number" && isFinite(col)) col = String(col);
      if (typeof col !== "string" || col === "" || seen[col]) return;
      seen[col] = true;
      out.push({col: col, dir: e.dir === "desc" ? "desc" : "asc"});
    });
    return out;
  }
  function defaultGet(it, col) {
    if (Array.isArray(it)) return it[col];
    if (col === "best") return it.best === undefined ? null : it.best;
    return it.keys && typeof it.keys === "object" ? it.keys[col] : it[col];
  }
  function order(items, keys, get) {
    get = get || defaultGet;
    keys = clean(keys);
    var rows = (items || []).map(function (it, i) {
      return {it: it, i: i, k: keys.map(function (k) { return get(it, k.col); })};
    });
    rows.sort(function (a, b) {
      for (var n = 0; n < keys.length; n++) {
        var c = compare(a.k[n], b.k[n], keys[n].col === "best" ? "asc" : keys[n].dir);
        if (c) return c;
      }
      return a.i - b.i;
    });
    return rows.map(function (r) { return r.it; });
  }
  function arrange(items, keys, get, role) {
    get = get || defaultGet;
    role = role || function (it) { return (it && it.role) || ""; };
    var sections = [{lead: null, rows: []}];
    (items || []).forEach(function (it) {
      if (role(it) === "group") sections.push({lead: it, rows: []});
      else sections[sections.length - 1].rows.push(it);
    });
    var all = [];
    sections.forEach(function (s) {
      if (s.lead) all.push(s.lead);
      arrangeSection(s.rows, keys, get, role).forEach(function (r) { all.push(r); });
    });
    return all;
  }
  // arrange() within one section (no "group" rows in `items`).
  function arrangeSection(items, keys, get, role) {
    var units = [], cur = null;
    (items || []).forEach(function (it) {
      var r = role(it);
      if (r === "follow" && cur) { cur.rows.push(it); return; }
      cur = {lead: it, role: r === "follow" ? "" : r, rows: [it]};
      units.push(cur);
    });
    var top = [], mid = [], last = [], bottom = [];
    units.forEach(function (u) { (u.role === "top" ? top : u.role === "bottom" ? bottom : u.role === "last" ? last : mid).push(u); });
    var byLead = function (u, col) { return get(u.lead, col); }, out = [];
    [top, order(mid, keys, byLead), order(last, keys, byLead), bottom].forEach(function (list) {
      list.forEach(function (u) { u.rows.forEach(function (r) { out.push(r); }); });
    });
    return out;
  }
  // ---- state ---------------------------------------------------------------------------------
  function parseState(s) {
    var v = s;
    if (typeof s === "string") { try { v = JSON.parse(s); } catch (e) { return null; } }
    if (v && typeof v === "object" && !Array.isArray(v)) v = [v];
    if (!Array.isArray(v)) return null;
    var out = clean(v);
    return out.length ? out : null;
  }
  function serialize(keys) { return JSON.stringify(clean(keys)); }
  // "col:dir,col" (dir defaults to asc), as `data-sort-default` holds it.
  function parseDefault(text) {
    if (!text) return [];
    return clean(String(text).split(",").map(function (p) {
      var t = p.trim(), at = t.lastIndexOf(":");
      if (at < 0) return {col: t, dir: "asc"};
      return {col: t.slice(0, at), dir: t.slice(at + 1)};
    }));
  }
  function same(a, b) {
    a = clean(a); b = clean(b);
    return a.length === b.length && a.every(function (k, i) { return k.col === b[i].col && k.dir === b[i].dir; });
  }
  function flip(d) { return d === "desc" ? "asc" : "desc"; }
  function click(keys, col, shift, first) {
    keys = clean(keys); col = String(col); first = first === "desc" ? "desc" : "asc";
    var at = -1;
    keys.forEach(function (k, i) { if (k.col === col) at = i; });
    if (shift) {
      if (at >= 0) return keys.map(function (k, i) { return i === at ? {col: col, dir: flip(k.dir)} : k; });
      return clean(keys.concat([{col: col, dir: first}]));
    }
    if (at === 0) return [{col: col, dir: flip(keys[0].dir)}];
    return [{col: col, dir: first}];
  }
  function dirWord(d) { return d === "desc" ? "descending" : "ascending"; }
  function describe(keys, label) {
    keys = clean(keys);
    label = label || function (c) { return c; };
    if (!keys.length) return "Default order";
    return "Sorted by " + keys.map(function (k) {
      return k.col === "best" ? "best first" : label(k.col) + ", " + dirWord(k.dir);
    }).join(", then by ");
  }
  root.PrismSort = {
    value: value, parse: parse, textOf: textOf, compare: compare, clean: clean, order: order, arrange: arrange,
    parseState: parseState, serialize: serialize, parseDefault: parseDefault, same: same, click: click, describe: describe
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
(function () {
  "use strict";
  if (typeof document === "undefined") return; // loaded by the node test for the pure part only
  var P = window.PrismSort;
  var PREFIX = "prism.sort:";

  // ---- storage (every access in try/catch; undefined = no storage, null = nothing stored) ------
  function store() { try { return window.localStorage || null; } catch (e) { return null; } }
  function storageKey(key) { return PREFIX + (location.pathname || "") + ":" + key; }
  function load(key) {
    try {
      var s = store();
      if (!s) return undefined;
      var raw = s.getItem(storageKey(key));
      return raw === null || raw === undefined ? null : P.parseState(raw);
    } catch (e) { return undefined; }
  }
  // True when stored (false: the choice lasts for this page).
  function save(key, keys) {
    try {
      var s = store();
      if (!s) return false;
      if (keys) s.setItem(storageKey(key), P.serialize(keys)); else s.removeItem(storageKey(key));
      return true;
    } catch (e) { return false; }
  }
  // ---- cells ---------------------------------------------------------------------------------
  // A form control's value as the server last showed it (its default), so a row does not move
  // while the user edits it.
  function controlValue(c) {
    if (c.type === "checkbox" || c.type === "radio") return c.defaultChecked ? 1 : 0;
    if (c.tagName === "SELECT") {
      var o = null;
      for (var i = 0; i < c.options.length && !o; i++) if (c.options[i].defaultSelected) o = c.options[i];
      o = o || c.options[0];
      return P.value({text: o ? o.textContent : ""});
    }
    return P.value({text: c.defaultValue});
  }
  function cellValue(td) {
    if (!td) return null;
    var v = td.getAttribute("data-v");
    if (v !== null) return P.value({v: v});
    var t = td.querySelector("time[datetime]");
    if (t) return P.value({datetime: t.getAttribute("datetime")});
    var text = P.textOf(td);
    if (!text) {
      var ctl = td.querySelector("input:not([type=hidden]), select, textarea");
      if (ctl) return controlValue(ctl);
    }
    return P.value({text: text});
  }
  // The cell over leaf column `x` (cells spanning columns count for each).
  function cellAt(tr, x) {
    var c = 0;
    for (var i = 0; i < tr.cells.length; i++) {
      var span = tr.cells[i].colSpan || 1;
      if (x < c + span) return tr.cells[i];
      c += span;
    }
    return null;
  }

  // ---- columns -------------------------------------------------------------------------------
  function labelOf(th) {
    if (th.dataset.sortLabel) return th.dataset.sortLabel;
    var out = [];
    (function walk(n) {
      Array.prototype.forEach.call(n.childNodes, function (c) {
        if (c.nodeType === 3) out.push(c.nodeValue);
        else if (c.nodeType === 1 && !/(^|\s)(th-sort|tag)(\s|$)/.test(c.getAttribute("class") || "")) walk(c);
      });
    })(th);
    return out.join("").replace(/\s+/g, " ").trim();
  }
  // Leaf columns of the head: [{index, th, col, label, first, on}].
  function columnsOf(table, S) {
    var head = table.tHead, grid = [];
    if (!head) return [];
    var rows = head.rows, n = rows.length, width = 0;
    for (var r = 0; r < n; r++) {
      grid[r] = grid[r] || [];
      var c = 0;
      Array.prototype.forEach.call(rows[r].cells, function (th) {
        while (grid[r][c]) c++;
        var span = th.colSpan || 1, down = th.rowSpan > 0 ? th.rowSpan : n - r;
        for (var dr = 0; dr < down && r + dr < n; dr++) {
          grid[r + dr] = grid[r + dr] || [];
          for (var dc = 0; dc < span; dc++) grid[r + dr][c + dc] = th;
        }
        c += span;
        width = Math.max(width, c);
      });
    }
    var names = (table.getAttribute("data-sort-cols") || "").split(",").map(function (s) { return s.trim(); });
    var cols = [];
    for (var x = 0; x < width; x++) {
      var th = grid[n - 1] && grid[n - 1][x];
      if (!th) continue;
      var parts = [];
      for (var y = 0; y < n; y++) {
        var h = grid[y][x];
        if (h && (y === 0 || h !== grid[y - 1][x])) { var t = labelOf(h); if (t) parts.push(t); }
      }
      var named = table.hasAttribute("data-sort-cols") ? names[x] : "";
      var col = th.dataset.sortCol || (named && named !== "-" ? named : String(x));
      var o = (S.columns && S.columns[col]) || {};
      var label = o.label || (th.dataset.sortLabel ? th.dataset.sortLabel : parts.join(" · "));
      var on = (th.colSpan || 1) === 1 && !!labelOf(th) && !th.hasAttribute("data-sort-off") && named !== "-" && !o.off;
      cols.push({index: x, th: th, col: col, label: label, first: o.first || th.dataset.sortFirst || "asc", on: on});
    }
    return cols;
  }

  // ---- the status line -------------------------------------------------------------------------
  var statusEl = null;
  function announce(text) {
    if (!document.body) return;
    if (!statusEl) {
      statusEl = document.createElement("div");
      statusEl.className = "sr-only"; statusEl.id = "prism-sort-status";
      statusEl.setAttribute("role", "status"); statusEl.setAttribute("aria-live", "polite");
      document.body.appendChild(statusEl);
    }
    // A new text node each time, so the same words twice are read twice.
    statusEl.textContent = "";
    window.setTimeout(function () { statusEl.textContent = text; }, 50);
  }

  // ---- state and changes ---------------------------------------------------------------------
  function effective(S) {
    var keys = S.keys || S.defaults;
    return keys.filter(function (k) { return k.col === "best" || S.byCol[k.col]; });
  }
  function labelFor(S) { return function (col) { var c = S.byCol[col]; return c ? c.label : col; }; }
  function defaultLabel(S) { return S.defaults.length && S.defaults[0].col === "best" ? "Best first" : "Default order"; }
  function change(table, keys, say) {
    var S = table._prismSort;
    keys = P.clean(keys);
    S.keys = P.same(keys, S.defaults) ? null : keys;
    S.unsaved = S.key ? !save(S.key, S.keys) : false;
    apply(table);
    if (say) announce(S.keys ? P.describe(effective(S), labelFor(S)) : defaultLabel(S));
    // Other tables of the same kind on this page follow.
    if (S.key) Array.prototype.forEach.call(document.querySelectorAll("table"), function (t) {
      if (t !== table && t._prismSort && t._prismSort.key === S.key) apply(t);
    });
  }

  // ---- the headers -----------------------------------------------------------------------------
  function onHeadClick(e) {
    var th = e.currentTarget, table = th.closest && th.closest("table"), S = table && table._prismSort;
    if (!S || th._prismSortCol === undefined) return;
    var t = e.target, ctl = t && t.closest ? t.closest("a, input, select, textarea, label, button, summary") : null;
    if (ctl && ctl !== th._prismSortBtn && th.contains(ctl)) return;
    e.preventDefault();
    var c = S.byCol[th._prismSortCol];
    change(table, P.click(effective(S), th._prismSortCol, !!e.shiftKey, c ? c.first : "asc"), true);
    var b = th._prismSortBtn;
    if (b && b.isConnected && document.activeElement !== b) { try { b.focus({preventScroll: true}); } catch (x) { b.focus(); } }
  }
  // Shift-click would select the header's text.
  function onHeadDown(e) { if (e.shiftKey) e.preventDefault(); }
  function decorate(table, S) {
    var cols = columnsOf(table, S);
    S.cols = cols.filter(function (c) { return c.on; });
    S.byCol = {}; S.index = {};
    S.cols.forEach(function (c) { S.byCol[c.col] = c; S.index[c.col] = c.index; });
    cols.forEach(function (c) {
      var th = c.th;
      if (!c.on) {
        if (th._prismSortBtn && th._prismSortBtn.parentNode === th) th.removeChild(th._prismSortBtn);
        th._prismSortCol = undefined; th.classList.remove("th-sortable");
        return;
      }
      th._prismSortCol = c.col;
      th.classList.add("th-sortable");
      if (!th._prismSortWired) {
        th._prismSortWired = true;
        th.addEventListener("click", onHeadClick);
        th.addEventListener("mousedown", onHeadDown);
      }
      var b = th._prismSortBtn;
      if (!b || !th.contains(b)) {
        b = document.createElement("button");
        b.type = "button"; b.className = "th-sort";
        var arrow = document.createElement("span"); arrow.className = "th-sort-arrow"; arrow.setAttribute("aria-hidden", "true");
        var rank = document.createElement("span"); rank.className = "th-sort-rank"; rank.setAttribute("aria-hidden", "true");
        b.appendChild(arrow); b.appendChild(rank);
        glue(th, b);
        th._prismSortBtn = b;
      }
      b.setAttribute("aria-label", "Sort by " + c.label);
      b.title = "Sort by " + c.label + " (Shift-click: add it as the next sort key)";
    });
  }
  // The button goes in a span with the header's last word (or its last element, such as a
  // glossary button): where the CSS makes .th-sort-glue nowrap (the Benchmark tables), a header
  // that wraps never leaves the button alone on a line (a word joiner does not hold before a
  // button in Chrome).
  function glue(th, b) {
    var g = document.createElement("span"), last = th.lastChild;
    g.className = "th-sort-glue";
    while (last && last.nodeType === 3 && !/\S/.test(last.nodeValue)) last = last.previousSibling;
    if (last && last.nodeType === 3) {
      var v = last.nodeValue.replace(/\s+$/, ""), i = v.search(/\S+$/);
      g.appendChild(document.createTextNode(v.slice(i)));
      if (i > 0) last.nodeValue = v.slice(0, i); else th.removeChild(last);
    } else if (last && last.nodeType === 1) g.appendChild(last);
    g.appendChild(b);
    th.appendChild(g);
  }
  function indicate(S, keys) {
    var rankOf = {};
    keys.forEach(function (k, i) { rankOf[k.col] = i; });
    S.cols.forEach(function (c) {
      var th = c.th, b = th._prismSortBtn, i = rankOf[c.col], on = i !== undefined, k = on ? keys[i] : null;
      if (on && i === 0) th.setAttribute("aria-sort", k.dir === "desc" ? "descending" : "ascending"); else th.removeAttribute("aria-sort");
      th.classList.toggle("th-sorted", on);
      if (!b) return;
      b.firstChild.textContent = !on ? "↕" : k.dir === "desc" ? "▼" : "▲";
      b.lastChild.textContent = on && keys.length > 1 ? String(i + 1) : "";
      b.classList.toggle("on", on);
    });
  }

  // ---- the rows --------------------------------------------------------------------------------
  function roleOf(tr) {
    if (tr.hasAttribute("data-sort-group")) return "group";
    if (tr.hasAttribute("data-sort-follow")) return "follow";
    var p = tr.getAttribute("data-sort-pin");
    if (p === "top" || p === "bottom") return p;
    return tr.classList.contains("lineup-out") ? "last" : "";
  }
  function sortRows(table, S, keys) {
    var get = function (tr, col) {
      if (col === "best") {
        var b = tr.getAttribute("data-best");
        return b === null || b.trim() === "" || !isFinite(Number(b)) ? null : Number(b);
      }
      var x = S.index[col];
      return x === undefined ? null : cellValue(cellAt(tr, x));
    };
    Array.prototype.forEach.call(table.tBodies, function (tb) {
      var rows = Array.prototype.slice.call(tb.rows);
      if (!rows.length) return;
      // Rows a render added take the next places of the table's own order.
      var max = -1;
      rows.forEach(function (r) { if (r._prismSortIdx !== undefined && r._prismSortIdx > max) max = r._prismSortIdx; });
      rows.forEach(function (r) { if (r._prismSortIdx === undefined) r._prismSortIdx = ++max; });
      var own = rows.slice().sort(function (a, b) { return a._prismSortIdx - b._prismSortIdx; });
      var out = P.arrange(own, keys, get, roleOf);
      if (out.every(function (r, i) { return r === rows[i]; })) return;
      var a = document.activeElement, keepFocus = a && a !== document.body && tb.contains(a);
      var frag = document.createDocumentFragment();
      out.forEach(function (r) { frag.appendChild(r); });
      tb.appendChild(frag);
      if (keepFocus && document.activeElement !== a && a.isConnected) { try { a.focus({preventScroll: true}); } catch (x) { a.focus(); } }
    });
  }

  // ---- the tools above the table (phones' Sort by, Default order) ------------------------------------
  function buildTools(table, S) {
    var box = document.createElement("div");
    box.className = "sort-tools";
    var l = document.createElement("label");
    l.className = "sort-by";
    var span = document.createElement("span");
    span.textContent = "Sort by ";
    var sel = document.createElement("select");
    sel.setAttribute("aria-label", "Sort the rows of this table by");
    sel.addEventListener("change", function () {
      var v = sel.value;
      if (v === "*") return;
      if (v === "") { change(table, S.defaults, true); return; }
      var at = v.lastIndexOf(":");
      change(table, [{col: v.slice(0, at), dir: v.slice(at + 1)}], true);
    });
    l.appendChild(span); l.appendChild(sel);
    var reset = document.createElement("button");
    reset.type = "button"; reset.className = "small secondary sort-reset";
    reset.title = "Back to this table's own order";
    reset.addEventListener("click", function () { change(table, S.defaults, true); });
    box.appendChild(l); box.appendChild(reset);
    S.tools = box; S.select = sel; S.reset = reset;
  }
  function option(sel, value, text) {
    var o = document.createElement("option");
    o.value = value; o.textContent = text;
    sel.appendChild(o);
    return o;
  }
  function fillTools(table, S, keys) {
    if (!S.tools) buildTools(table, S);
    // A choice the select cannot name (several keys, or `best`) is one more, disabled, option.
    var sel = S.select, dflt = defaultLabel(S), custom = !!S.keys && keys.length > 0 && (keys.length > 1 || !S.byCol[keys[0].col]);
    var sig = JSON.stringify([dflt, S.cols.map(function (c) { return [c.col, c.label]; }), custom ? P.describe(keys, labelFor(S)) : ""]);
    if (S.toolsSig !== sig) {
      S.toolsSig = sig;
      sel.replaceChildren();
      option(sel, "", dflt);
      S.cols.forEach(function (c) {
        option(sel, c.col + ":asc", c.label + ", ascending");
        option(sel, c.col + ":desc", c.label + ", descending");
      });
      if (custom) option(sel, "*", P.describe(keys, labelFor(S))).disabled = true;
    }
    sel.value = !S.keys || !keys.length ? "" : custom ? "*" : keys[0].col + ":" + keys[0].dir;
    S.reset.textContent = dflt;
    S.reset.hidden = !S.keys;
    S.tools.classList.toggle("sorted", !!S.keys);
    S.tools.hidden = !S.cols.length;
  }
  function placeTools(table, S) {
    if (S.tools.isConnected && S.tools.nextElementSibling && (S.tools.nextElementSibling === table || S.tools.nextElementSibling.contains(table))) return;
    var p = table.parentNode, anchor = table;
    if (p && p.nodeType === 1 && p.classList.contains("scroll") && p.firstElementChild === table) anchor = p;
    if (anchor.parentNode) anchor.parentNode.insertBefore(S.tools, anchor);
  }

  // ---- attach / apply --------------------------------------------------------------------------
  function attach(table, o) {
    if (!table || table.tagName !== "TABLE") return null;
    o = o || {};
    var S = table._prismSort;
    if (S) {
      if (o.defaultKeys) S.defaults = P.clean(o.defaultKeys);
      if (o.columns) S.columns = o.columns;
      apply(table);
      return S;
    }
    var key = o.key || table.getAttribute("data-sort") || null;
    S = table._prismSort = {
      key: key,
      defaults: o.defaultKeys ? P.clean(o.defaultKeys) : P.parseDefault(table.getAttribute("data-sort-default")),
      columns: o.columns || {}, keys: null, cols: [], byCol: {}, index: {}, tools: null
    };
    if (key) { var st = load(key); if (st) S.keys = P.same(st, S.defaults) ? null : st; }
    apply(table);
    // Rows and the wrapper a builder adds right after attaching: sorted (and the tools placed) once
    // it is done.
    Promise.resolve().then(function () { apply(table); });
    return S;
  }
  function apply(table) {
    if (!table || table.tagName !== "TABLE") return;
    var S = table._prismSort;
    if (!S) { if (table.hasAttribute("data-sort")) attach(table); return; }
    if (S.key && !S.unsaved) { var st = load(S.key); if (st !== undefined) S.keys = st && !P.same(st, S.defaults) ? st : null; }
    decorate(table, S);
    var keys = effective(S);
    sortRows(table, S, keys);
    indicate(S, keys);
    fillTools(table, S, keys);
    placeTools(table, S);
  }
  // The keys a table shows now (null: its own order); restore() puts them back on a rebuilt one.
  function state(table) {
    var S = table && table._prismSort;
    return S ? (S.keys ? P.clean(S.keys) : null) : undefined;
  }
  function restore(table, keys) {
    if (!table || table.tagName !== "TABLE") return;
    var S = table._prismSort;
    if (!S) { if (!table.hasAttribute("data-sort")) return; S = attach(table); }
    // A stored state is the truth; without storage (or a key) the captured one is.
    if (keys !== undefined && (!S.key || load(S.key) === undefined)) S.keys = keys && !P.same(keys, S.defaults) ? P.clean(keys) : null;
    apply(table);
  }
  function scan(rootEl) {
    var r = rootEl || document;
    if (!r.querySelectorAll) return;
    Array.prototype.forEach.call(r.querySelectorAll("table[data-sort]"), function (t) { if (!t._prismSort) attach(t); });
  }
  P.attach = attach; P.apply = apply; P.state = state; P.restore = restore; P.scan = scan;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", function () { scan(document); });
  else scan(document);
})();
