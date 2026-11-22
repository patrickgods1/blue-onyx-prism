//! The table sorter of every page, `assets/sort.js` (`PrismSort`), its pure part run in `node`:
//! the sort value of a cell (numbers, units, ratios, dashes, times, `data-v`), a cell's text, the
//! stable multi-key order with empties last in both directions, the `best` pseudo-column, row
//! groups (detail rows that follow, pinned and "last" rows, `data-sort-group` sections), what a
//! click or Shift-click does, and the stored state, including both old shapes of the Benchmark
//! page; then the page part on a small stub DOM (headers, clicks, the phones' select, Default
//! order, re-renders, migration, sections).
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

fn have_node() -> bool {
    let ok = Command::new("node")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        assert!(
            std::env::var("BOP_REQUIRE_NODE").as_deref() != Ok("1"),
            "BOP_REQUIRE_NODE=1 but `node` does not run: install Node or unset the variable"
        );
        println!("skipped: no node");
    }
    ok
}

/// Run `body` after loading `assets/sort.js` (no `document`: the pure part only) as `S`; it
/// asserts and exits non-zero on a failure.
fn run(body: &str) {
    node(&format!(
        r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({{console}});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const S = vm.runInContext('PrismSort', ctx);
// Plain copies: objects made in the context have that context's prototypes.
const plain = (x) => JSON.parse(JSON.stringify(x));
{body}
console.log('ok');
"#
    ));
}

/// Run `driver_js` with the path of `assets/sort.js` as its argument.
fn node(driver_js: &str) {
    let dir = std::env::temp_dir().join(format!("bop-js-sort-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let driver = dir.join("driver.js");
    std::fs::write(&driver, driver_js).unwrap();
    let asset = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/sort.js");
    let out = Command::new("node")
        .arg(&driver)
        .arg(asset)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "node failed ({}): {}{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("ok"));
}

#[test]
fn a_cell_value_reads_numbers_units_ratios_dashes_and_times() {
    if !have_node() {
        return;
    }
    run(r#"
const v = (text) => S.value({text});
// Numbers: thousands separators, signs (also the minus sign), decimals.
assert.strictEqual(v('42'), 42);
assert.strictEqual(v(' 1,234,567 '), 1234567);
assert.strictEqual(v('-3.5'), -3.5);
assert.strictEqual(v('+7'), 7);
assert.strictEqual(v('−2.5'), -2.5);
assert.strictEqual(v('.5'), 0.5);
assert.strictEqual(v('1e3'), 1000);
// Percent and units: ms and s on one scale, decimal sizes, px.
assert.strictEqual(v('45.3%'), 45.3);
assert.strictEqual(v('45.3 %'), 45.3);
assert.strictEqual(v('120 ms'), 120);
assert.strictEqual(v('1.5 s'), 1500);
assert.ok(v('1.5 s') > v('999 ms'), 'seconds and milliseconds compare');
assert.strictEqual(v('2 kB'), 2000);
assert.strictEqual(v('83 MB'), 83e6);
assert.strictEqual(v('1.7 GB'), 1.7e9);
assert.ok(v('1.0 GB') > v('999 MB'), 'GB and MB compare');
assert.strictEqual(v('512 B'), 512);
assert.strictEqual(v('640 px'), 640);
assert.strictEqual(v('12.5/s'), 12.5);
// a/b is the ratio; a zero denominator has none.
assert.strictEqual(v('3/4'), 0.75);
assert.strictEqual(v('0/16'), 0);
assert.strictEqual(v('2/0'), null);
// With spaces it is text ("45.3 / 2.1" is two rates, not a ratio).
assert.strictEqual(v('45.3 / 2.1'), '45.3 / 2.1');
// Dashes, n/a and nothing are empty.
for (const e of ['', '   ', '-', '–', '—', 'n/a', 'N/A', null, undefined]) assert.strictEqual(v(e), null, JSON.stringify(e));
// Anything else is lowercased text.
assert.strictEqual(v('Yolo26S'), 'yolo26s');
assert.strictEqual(v('12 (3 confirmed)'), '12 (3 confirmed)');
// data-v wins over the text: numeric or a string; empty-like is empty.
assert.strictEqual(S.value({v: '-0.75', text: 'whatever'}), -0.75);
assert.strictEqual(S.value({v: 3, text: ''}), 3);
assert.strictEqual(S.value({v: 'GPU.0'}), 'gpu.0');
assert.strictEqual(S.value({v: ''}), null);
assert.strictEqual(S.value({v: '-'}), null);
assert.strictEqual(S.value({v: 'Infinity'}), 'infinity');
// A <time datetime> next: epoch milliseconds.
assert.strictEqual(S.value({datetime: '2026-11-18T10:00:00Z', text: '18 Nov'}), Date.parse('2026-11-18T10:00:00Z'));
assert.ok(S.value({datetime: '2026-11-18T10:00:00Z'}) > S.value({datetime: '2026-11-17T23:59:59Z'}));
assert.strictEqual(S.value({datetime: 'not a time'}), 'not a time');
assert.strictEqual(S.value(null), null);

// A cell's text: its first line, without badges, sub lines, hidden parts, controls and action
// buttons; a term button holds data.
const T = (t) => ({nodeType: 3, nodeValue: t});
const E = (name, cls, kids, more) => Object.assign({nodeType: 1, nodeName: name, className: cls || '', childNodes: kids || []}, more || {});
assert.strictEqual(S.textOf(E('TD', '', [T('ipcam-general '), E('SPAN', 'tag', [T('default')])])), 'ipcam-general');
assert.strictEqual(S.textOf(E('TD', '', [T('12.3'), E('SPAN', 'sub', [T('10–15')])])), '12.3');
assert.strictEqual(S.textOf(E('TD', '', [T('Failed:'), E('BR'), E('SPAN', 'detail', [T('model file not found')])])), 'Failed:');
assert.strictEqual(S.textOf(E('TD', '', [E('CODE', '', [T('imp-1')]), E('BR'), E('SPAN', 'muted small', [T('yesterday')])])), 'imp-1');
assert.strictEqual(S.textOf(E('TD', '', [E('BR'), T('after an empty first line')])), 'after an empty first line');
assert.strictEqual(S.textOf(E('TD', '', [T('a '), E('BUTTON', 'small', [T('Remove')]), E('BUTTON', 'term', [T('b')])])), 'a b');
assert.strictEqual(S.textOf(E('TD', '', [T('x'), E('SPAN', '', [T('gone')], {hidden: true}), E('SELECT', '', [E('OPTION', '', [T('opt')])]), E('DETAILS', '', [T('more')])])), 'x');
assert.strictEqual(S.textOf(null), '');
"#);
}

#[test]
fn the_order_is_stable_with_empties_last_in_both_directions() {
    if !have_node() {
        return;
    }
    run(r#"
const names = (rows) => plain(rows.map((r) => r.name));
const rows = [
  {name: 'a', keys: {0: 3, 1: 'x'}},
  {name: 'b', keys: {0: null, 1: 'y'}},
  {name: 'c', keys: {0: 1, 1: 'x'}},
  {name: 'd', keys: {0: 'text', 1: 'y'}},
  {name: 'e', keys: {0: 3, 1: 'w'}},
  {name: 'f', keys: {0: 2, 1: null}},
];
// Ascending: numbers, then strings, then empties; equal values keep their order (a before e).
assert.deepStrictEqual(names(S.order(rows, [{col: '0', dir: 'asc'}])), ['c', 'f', 'a', 'e', 'd', 'b']);
// Descending turns numbers and strings, never where the empties go.
assert.deepStrictEqual(names(S.order(rows, [{col: '0', dir: 'desc'}])), ['a', 'e', 'f', 'c', 'd', 'b']);
// A second key decides the ties of the first.
assert.deepStrictEqual(names(S.order(rows, [{col: '0', dir: 'desc'}, {col: '1', dir: 'asc'}])), ['e', 'a', 'f', 'c', 'd', 'b']);
assert.deepStrictEqual(names(S.order(rows, [{col: '1', dir: 'asc'}, {col: '0', dir: 'asc'}])), ['e', 'c', 'a', 'd', 'b', 'f']);
assert.deepStrictEqual(names(S.order(rows, [{col: '1', dir: 'desc'}, {col: '0', dir: 'asc'}])), ['d', 'b', 'c', 'a', 'e', 'f']);
// No keys: the order given.
assert.deepStrictEqual(names(S.order(rows, [])), ['a', 'b', 'c', 'd', 'e', 'f']);
// Text compares naturally and without case.
const cams = [{name: 'cam10', keys: {0: 'cam10'}}, {name: 'cam2', keys: {0: 'cam2'}}, {name: 'Cam1', keys: {0: 'cam1'}}];
assert.deepStrictEqual(names(S.order(cams, [{col: '0', dir: 'asc'}])), ['Cam1', 'cam2', 'cam10']);
// Arrays of keys work too (col names are strings of the index).
assert.deepStrictEqual(plain(S.order([[2], [1], [3]], [{col: '0', dir: 'asc'}]).map((r) => r[0])), [1, 2, 3]);
// An accessor.
const byLen = S.order(['ccc', 'a', 'bb'], [{col: 'len', dir: 'desc'}], (s) => s.length);
assert.deepStrictEqual(plain(byLen), ['ccc', 'bb', 'a']);
// Many equal rows stay in their order (the sort is stable whatever the engine does).
const many = Array.from({length: 50}, (_, i) => ({i, keys: {0: i % 3 === 0 ? null : 1}}));
const sorted = S.order(many, [{col: '0', dir: 'desc'}]);
const ones = sorted.filter((r) => r.keys[0] === 1).map((r) => r.i), nones = sorted.filter((r) => r.keys[0] === null).map((r) => r.i);
assert.deepStrictEqual(ones, ones.slice().sort((a, b) => a - b));
assert.deepStrictEqual(nones, nones.slice().sort((a, b) => a - b));
assert.deepStrictEqual(sorted.slice(-nones.length).map((r) => r.i), nones, 'empties last');
// compare() alone.
assert.ok(S.compare(1, 2, 'asc') < 0 && S.compare(1, 2, 'desc') > 0);
assert.ok(S.compare(null, 1, 'asc') > 0 && S.compare(null, 1, 'desc') > 0);
assert.ok(S.compare(5, 'a', 'asc') < 0 && S.compare(5, 'a', 'desc') < 0);
assert.strictEqual(S.compare(NaN, null, 'asc'), 0);
"#);
}

#[test]
fn best_first_and_row_groups() {
    if !have_node() {
        return;
    }
    run(r#"
const names = (rows) => plain(rows.map((r) => r.name));
// `best`: the row's ordinal, ascending whatever the dir; rows without one after, in order.
const ranked = [
  {name: 'a', best: null, keys: {0: 1}},
  {name: 'b', best: 2, keys: {0: 2}},
  {name: 'c', keys: {0: 3}},
  {name: 'd', best: 1, keys: {0: 4}},
  {name: 'e', best: 3, keys: {0: 5}},
];
assert.deepStrictEqual(names(S.order(ranked, [{col: 'best', dir: 'asc'}])), ['d', 'b', 'e', 'a', 'c']);
assert.deepStrictEqual(names(S.order(ranked, [{col: 'best', dir: 'desc'}])), ['d', 'b', 'e', 'a', 'c']);
// The rows without a rank by a second key.
assert.deepStrictEqual(names(S.order(ranked, [{col: 'best', dir: 'asc'}, {col: '0', dir: 'desc'}])), ['d', 'b', 'e', 'c', 'a']);

// Groups: a detail row stays behind its lead, pinned rows stay at their end, "last" rows sort
// among themselves after the others.
const rows = [
  {name: 'zero', role: 'top', keys: {0: 100}},
  {name: 'r1', keys: {0: 2}},
  {name: 'r1-switch', role: 'follow', keys: {0: 999}},
  {name: 'r1-metrics', role: 'follow', keys: {0: -5}},
  {name: 'r2', keys: {0: 1}},
  {name: 'r2-metrics', role: 'follow'},
  {name: 'out1', role: 'last', keys: {0: 9}},
  {name: 'r3', keys: {0: 3}},
  {name: 'out2', role: 'last', keys: {0: 8}},
  {name: 'total', role: 'bottom', keys: {0: -1}},
  {name: 'total-detail', role: 'follow'},
];
assert.deepStrictEqual(names(S.arrange(rows, [{col: '0', dir: 'asc'}])),
  ['zero', 'r2', 'r2-metrics', 'r1', 'r1-switch', 'r1-metrics', 'r3', 'out2', 'out1', 'total', 'total-detail']);
assert.deepStrictEqual(names(S.arrange(rows, [{col: '0', dir: 'desc'}])),
  ['zero', 'r3', 'r1', 'r1-switch', 'r1-metrics', 'r2', 'r2-metrics', 'out1', 'out2', 'total', 'total-detail']);
// No keys: as given (the "last" rows still after the others).
assert.deepStrictEqual(names(S.arrange(rows, [])),
  ['zero', 'r1', 'r1-switch', 'r1-metrics', 'r2', 'r2-metrics', 'r3', 'out1', 'out2', 'total', 'total-detail']);
// A detail row with no row before it leads its own group.
assert.deepStrictEqual(names(S.arrange([{name: 'x', role: 'follow', keys: {0: 2}}, {name: 'y', keys: {0: 1}}], [{col: '0', dir: 'asc'}])), ['y', 'x']);
// A role accessor and a value accessor (as the page passes them).
const trs = [{id: 1, v: 5}, {id: 2, v: 0, f: true}, {id: 3, v: 1}];
const out = S.arrange(trs, [{col: 'v', dir: 'asc'}], (r, c) => r[c], (r) => r.f ? 'follow' : '');
assert.deepStrictEqual(plain(out.map((r) => r.id)), [3, 1, 2]);

// Sections (`data-sort-group`): rows sort only among the rows of their section, the sections and
// their lead rows keep their places, roles work inside each; rows before the first lead are a
// section of their own.
const grouped = [
  {name: 'loose', keys: {0: 7}},
  {name: 'g1', role: 'group', best: 1, keys: {0: 0}},
  {name: 'a', best: 2, keys: {0: 3}},
  {name: 'a-detail', role: 'follow'},
  {name: 'b', best: 1, keys: {0: 1}},
  {name: 'b-detail', role: 'follow', keys: {0: 100}},
  {name: 'g2', role: 'group', keys: {0: 99}},
  {name: 'c', keys: {0: 9}},
  {name: 'pinned', role: 'top', keys: {0: 50}},
  {name: 'd', best: 1, keys: {0: 2}},
  {name: 'g3', role: 'group'},
];
assert.deepStrictEqual(names(S.arrange(grouped, [{col: 'best', dir: 'asc'}])),
  ['loose', 'g1', 'b', 'b-detail', 'a', 'a-detail', 'g2', 'pinned', 'd', 'c', 'g3']);
assert.deepStrictEqual(names(S.arrange(grouped, [{col: '0', dir: 'desc'}])),
  ['loose', 'g1', 'a', 'a-detail', 'b', 'b-detail', 'g2', 'pinned', 'c', 'd', 'g3']);
// No keys: as given, but for the pinned row at the top of its own section.
assert.deepStrictEqual(names(S.arrange(grouped, [])),
  ['loose', 'g1', 'a', 'a-detail', 'b', 'b-detail', 'g2', 'pinned', 'c', 'd', 'g3']);
"#);
}

#[test]
fn clicks_and_the_stored_state() {
    if !have_node() {
        return;
    }
    run(r#"
// Plain click: that column alone; again: the other direction. Shift-click: the next key, or
// that key flipped where it is.
let k = S.click([], 'name', false);
assert.deepStrictEqual(plain(k), [{col: 'name', dir: 'asc'}]);
k = S.click(k, 'name', false);
assert.deepStrictEqual(plain(k), [{col: 'name', dir: 'desc'}]);
k = S.click(k, 'fps', true, 'desc');
assert.deepStrictEqual(plain(k), [{col: 'name', dir: 'desc'}, {col: 'fps', dir: 'desc'}]);
k = S.click(k, 'fps', true);
assert.deepStrictEqual(plain(k), [{col: 'name', dir: 'desc'}, {col: 'fps', dir: 'asc'}]);
k = S.click(k, 'name', true);
assert.deepStrictEqual(plain(k), [{col: 'name', dir: 'asc'}, {col: 'fps', dir: 'asc'}]);
assert.deepStrictEqual(plain(S.click(k, 'fps', false)), [{col: 'fps', dir: 'asc'}], 'a plain click on a second key: that column alone');
assert.deepStrictEqual(plain(S.click(k, 'name', false)), [{col: 'name', dir: 'desc'}], 'a plain click on the first key flips it');
assert.deepStrictEqual(plain(S.click([{col: 'best', dir: 'asc'}], 'ms', false, 'desc')), [{col: 'ms', dir: 'desc'}]);
assert.deepStrictEqual(plain(S.click([], 3, false)), [{col: '3', dir: 'asc'}], 'index columns are strings');

// The stored state: the list, and both old Benchmark shapes.
assert.deepStrictEqual(plain(S.parseState('[{"col":"a","dir":"desc"},{"col":"b","dir":"asc"}]')), [{col: 'a', dir: 'desc'}, {col: 'b', dir: 'asc'}]);
assert.deepStrictEqual(plain(S.parseState('{"col":"model","dir":"desc"}')), [{col: 'model', dir: 'desc'}]);
// Nothing usable: null (the table's own order).
for (const bad of [null, undefined, '', 'not json', '42', '"text"', '[]', '{}', '{"sort":null,"q":"x"}', '{"q":"x"}', '[{"dir":"asc"}]', '[{"col":""}]']) {
  assert.strictEqual(S.parseState(bad), null, String(bad));
}
// Cleaned: a bad dir is ascending, a numeric col a string, duplicates and junk dropped, at most 8.
assert.deepStrictEqual(plain(S.parseState('[{"col":"a","dir":"sideways"},{"col":2,"dir":"desc"},{"col":"a","dir":"desc"},null,7,{"col":{}}]')), [{col: 'a', dir: 'asc'}, {col: '2', dir: 'desc'}]);
const lots = Array.from({length: 12}, (_, i) => ({col: 'c' + i, dir: 'asc'}));
assert.strictEqual(S.parseState(JSON.stringify(lots)).length, 8);
// serialize is its inverse.
const keys = [{col: 'a', dir: 'desc'}, {col: 'best', dir: 'asc'}];
assert.deepStrictEqual(plain(S.parseState(S.serialize(keys))), keys);
assert.strictEqual(S.serialize([{col: 'x', dir: 'desc', junk: 1}]), '[{"col":"x","dir":"desc"}]');
// The data-sort-default attribute.
assert.deepStrictEqual(plain(S.parseDefault('best')), [{col: 'best', dir: 'asc'}]);
assert.deepStrictEqual(plain(S.parseDefault('model:desc, 3')), [{col: 'model', dir: 'desc'}, {col: '3', dir: 'asc'}]);
assert.deepStrictEqual(plain(S.parseDefault('')), []);
assert.ok(S.same([{col: 'a', dir: 'asc'}], [{col: 'a'}]) && !S.same([{col: 'a', dir: 'asc'}], [{col: 'a', dir: 'desc'}]) && S.same([], null));
// What the status line says.
assert.strictEqual(S.describe([{col: 'fps', dir: 'desc'}, {col: 'model', dir: 'asc'}], (c) => ({fps: 'Speed', model: 'Model'})[c]),
  'Sorted by Speed, descending, then by Model, ascending');
assert.strictEqual(S.describe([]), 'Default order');
assert.strictEqual(S.describe([{col: 'best', dir: 'asc'}]), 'Sorted by best first');
"#);
}

/// The page part on a small stub DOM: a server table opted in by `data-sort`, a click, a second
/// click, Shift-click ranks, a term button in a header, the phones' select and Default order, rows
/// a poll adds, a table built detached and filled before the microtask, detail and pinned rows,
/// two header rows with a group, the move of an old Benchmark sort, state / restore of a table
/// without a key, and the Best first label.
const PAGE: &str = r#"
// A small DOM: enough of Node / Element / table / select for sort.js, no layout.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const src = fs.readFileSync(process.argv[2], 'utf8');

function camel(s) { return s.replace(/-([a-z])/g, (_, c) => c.toUpperCase()); }
function kebab(s) { return s.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase()); }
let doc;
class Node {
  constructor(type) { this.nodeType = type; this.childNodes = []; this.parentNode = null; }
  get isConnected() { let n = this; while (n.parentNode) n = n.parentNode; return n === doc; }
  appendChild(c) { if (c.nodeType === 11) { c.childNodes.slice().forEach((k) => this.appendChild(k)); return c; } if (c.parentNode) c.parentNode.removeChild(c); this.childNodes.push(c); c.parentNode = this; return c; }
  insertBefore(c, ref) { if (!ref) return this.appendChild(c); if (c.parentNode) c.parentNode.removeChild(c); const i = this.childNodes.indexOf(ref); this.childNodes.splice(i, 0, c); c.parentNode = this; return c; }
  removeChild(c) { const i = this.childNodes.indexOf(c); if (i >= 0) this.childNodes.splice(i, 1); c.parentNode = null; return c; }
  replaceChildren(...kids) { this.childNodes.slice().forEach((k) => this.removeChild(k)); kids.forEach((k) => this.appendChild(k)); }
  contains(n) { while (n) { if (n === this) return true; n = n.parentNode; } return false; }
  get textContent() { return this.nodeType === 3 ? this.nodeValue : this.childNodes.map((c) => c.textContent).join(''); }
  set textContent(t) { this.childNodes.slice().forEach((k) => this.removeChild(k)); if (t !== '' && t != null) this.appendChild(new Text(String(t))); }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  get previousSibling() { const p = this.parentNode; if (!p) return null; const i = p.childNodes.indexOf(this); return i > 0 ? p.childNodes[i - 1] : null; }
}
class Text extends Node { constructor(t) { super(3); this.nodeValue = t; this.nodeName = '#text'; } }
class Frag extends Node { constructor() { super(11); } }
class El extends Node {
  constructor(tag) {
    super(1); this.tagName = tag.toUpperCase(); this.nodeName = this.tagName; this.attrs = {}; this.listeners = {};
    const self = this;
    this.dataset = new Proxy({}, {
      get: (_, k) => self.attrs['data-' + kebab(String(k))],
      set: (_, k, v) => { self.attrs['data-' + kebab(String(k))] = String(v); return true; },
      deleteProperty: (_, k) => { delete self.attrs['data-' + kebab(String(k))]; return true; },
    });
    this.classList = {
      add: (c) => { const s = new Set(self.className.split(/\s+/).filter(Boolean)); s.add(c); self.className = [...s].join(' '); },
      remove: (c) => { self.className = self.className.split(/\s+/).filter((x) => x && x !== c).join(' '); },
      contains: (c) => self.className.split(/\s+/).includes(c),
      toggle: (c, on) => { if (on === undefined) on = !self.classList.contains(c); if (on) self.classList.add(c); else self.classList.remove(c); return on; },
    };
    this.hidden = false; this.colSpan = 1; this.rowSpan = 1;
  }
  get className() { return this.attrs['class'] || ''; }
  set className(v) { this.attrs['class'] = v; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  hasAttribute(k) { return k in this.attrs; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
  get firstElementChild() { return this.children[0] || null; }
  get nextElementSibling() { if (!this.parentNode) return null; const s = this.parentNode.children; return s[s.indexOf(this) + 1] || null; }
  matches(sel) { return matches(this, sel); }
  closest(sel) { let n = this; while (n && n.nodeType === 1) { if (matches(n, sel)) return n; n = n.parentNode; } return null; }
  querySelectorAll(sel) { const out = []; const walk = (n) => n.children.forEach((c) => { if (matches(c, sel)) out.push(c); walk(c); }); walk(this); return out; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  focus() { doc.activeElement = this; }
  get tHead() { return this.children.find((c) => c.tagName === 'THEAD') || null; }
  get tBodies() { return this.children.filter((c) => c.tagName === 'TBODY'); }
  get rows() { return this.children.filter((c) => c.tagName === 'TR'); }
  get cells() { return this.children.filter((c) => c.tagName === 'TD' || c.tagName === 'TH'); }
  get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
  get value() { if (this.tagName === 'SELECT') { const o = this.options.find((x) => x.selected) || this.options[0]; return o ? o.value : ''; } return this._value || ''; }
  set value(v) { if (this.tagName === 'SELECT') { this.options.forEach((o) => { o.selected = o.value === v; }); } else this._value = v; }
  get selectedIndex() { return this.options.findIndex((o) => o.selected); }
  dispatch(type, ev) { ev = Object.assign({type, target: this, currentTarget: this, preventDefault() { this.defaultPrevented = true; }, shiftKey: false}, ev || {}); (this.listeners[type] || []).forEach((f) => f.call(this, ev)); return ev; }
}
// selectors: tag, .class, [attr], [attr="v"], comma lists, `a, b`, `:not([type=hidden])` ignored
function matches(el, sel) {
  return sel.split(',').some((s) => {
    s = s.trim().replace(/:not\([^)]*\)/g, '');
    const m = /^([a-z0-9]*)((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i.exec(s);
    if (!m) throw new Error('selector ' + s);
    if (m[1] && el.tagName !== m[1].toUpperCase()) return false;
    for (const c of (m[2].match(/\.[\w-]+/g) || [])) if (!el.classList.contains(c.slice(1))) return false;
    for (const a of (m[3].match(/\[[^\]]+\]/g) || [])) {
      const mm = /^\[([\w-]+)(?:="?([^"\]]*)"?)?\]$/.exec(a);
      if (!el.hasAttribute(mm[1])) return false;
      if (mm[2] !== undefined && el.getAttribute(mm[1]) !== mm[2]) return false;
    }
    return true;
  });
}
const store = {};
doc = new El('#document');
doc.nodeType = 9;
doc.body = new El('body'); doc.appendChild(doc.body);
doc.readyState = 'complete';
doc.activeElement = doc.body;
doc.createElement = (t) => new El(t);
doc.createDocumentFragment = () => new Frag();
doc.createTextNode = (t) => new Text(String(t));
doc.addEventListener = () => {};
const win = {document: doc, location: {pathname: '/stats'}, console, Promise, setTimeout: (f) => f(),
  localStorage: {getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: (k) => { delete store[k]; }}};
win.window = win;
const ctx = vm.createContext(win);
vm.runInContext(src, ctx);
const S = vm.runInContext('PrismSort', ctx);

function el(tag, attrs, kids) { const e = new El(tag); Object.entries(attrs || {}).forEach(([k, v]) => k === 'text' ? e.appendChild(new Text(v)) : e.setAttribute(k, v)); (kids || []).forEach((k) => e.appendChild(typeof k === 'string' ? new Text(k) : k)); return e; }
function names(t) { return t.tBodies[0].rows.map((r) => r.cells[0].textContent); }

// A Stats-like table: Model, Requests, Queue, pending (off).
const thead = el('thead', {}, [el('tr', {}, [el('th', {text: 'Model'}), el('th', {text: 'Requests'}), el('th', {text: 'Queue'}), el('th', {'data-sort-off': ''}, [el('span', {class: 'sr-only', text: 'Pending'})])])]);
const tb = el('tbody', {}, [
  el('tr', {'data-model': 'b'}, [el('td', {text: 'b'}), el('td', {text: '1,200'}), el('td', {text: '2/16'}), el('td')]),
  el('tr', {'data-model': 'a'}, [el('td', {}, ['a ', el('span', {class: 'tag', text: 'default'})]), el('td', {text: '30'}), el('td', {text: '0/16'}), el('td')]),
  el('tr', {'data-model': 'c'}, [el('td', {text: 'c'}), el('td', {text: '-'}), el('td', {text: '8/16'}), el('td')]),
]);
const table = el('table', {'data-sort': 'models'}, [thead, tb]);
const wrap = el('div', {class: 'scroll'}, [table]);
doc.body.appendChild(wrap);
S.scan(doc);
assert.ok(table._prismSort, 'attached by scan');
const ths = thead.rows[0].cells;
assert.ok(ths[0].querySelector('button.th-sort'), 'button in Model');
assert.ok(!ths[3].querySelector('button.th-sort'), 'no button in the off column');
assert.ok(doc.body.children[0].classList.contains('sort-tools'), 'tools before the scroll wrapper');
assert.deepStrictEqual(names(table), ['b', 'a default', 'c'], 'own order');
// Click Requests: asc, empties last.
ths[1].dispatch('click', {target: ths[1]});
assert.deepStrictEqual(names(table), ['a default', 'b', 'c']);
assert.strictEqual(ths[1].getAttribute('aria-sort'), 'ascending');
assert.strictEqual(store['prism.sort:/stats:models'], '[{"col":"1","dir":"asc"}]');
ths[1].dispatch('click', {target: ths[1]});
assert.deepStrictEqual(names(table), ['b', 'a default', 'c'], 'desc, empty still last');
assert.strictEqual(ths[1].getAttribute('aria-sort'), 'descending');
// Shift-click Queue: second key, rank numbers.
ths[2].dispatch('click', {target: ths[2], shiftKey: true});
assert.strictEqual(ths[1].querySelector('.th-sort-rank').textContent, '1');
assert.strictEqual(ths[2].querySelector('.th-sort-rank').textContent, '2');
assert.strictEqual(ths[2].getAttribute('aria-sort'), null);
// A click on a term button in the header does nothing (here: any other button).
const before = store['prism.sort:/stats:models'];
const term = el('button', {class: 'term', text: '?'}); ths[0].insertBefore(term, ths[0].firstChild);
ths[0].dispatch('click', {target: term});
assert.strictEqual(store['prism.sort:/stats:models'], before, 'term click does not sort');
// The select: options and value.
const tools = doc.body.children[0], sel = tools.querySelector('select'), reset = tools.querySelector('button');
assert.strictEqual(sel.value, '*', 'custom (two keys)');
assert.strictEqual(reset.hidden, false);
assert.strictEqual(reset.textContent, 'Default order');
sel.value = '0:desc'; sel.dispatch('change');
assert.deepStrictEqual(names(table), ['c', 'b', 'a default'], 'model desc');
assert.strictEqual(sel.value, '0:desc');
// A poll re-renders in place: a new row and changed values; apply keeps the sort.
tb.appendChild(el('tr', {'data-model': 'd'}, [el('td', {text: 'd'}), el('td', {text: '5'}), el('td', {text: '1/16'}), el('td')]));
S.apply(table);
assert.deepStrictEqual(names(table), ['d', 'c', 'b', 'a default']);
// Default order: back to the own order (d last: added after).
reset.dispatch('click');
assert.deepStrictEqual(names(table), ['b', 'a default', 'c', 'd']);
assert.strictEqual(store['prism.sort:/stats:models'], undefined, 'default is not stored');
assert.strictEqual(reset.hidden, true);
// A table built later with the same key, attached detached: sorted once filled, tools placed.
ths[2].dispatch('click', {target: ths[2]});
const t2 = el('table', {}, [el('thead', {}, [el('tr', {}, [el('th', {text: 'M'}), el('th', {text: 'R'}), el('th', {text: 'Q'})])]), el('tbody')]);
S.attach(t2, {key: 'models'});
[['x', '1', '3/4'], ['y', '2', '1/4'], ['z', '3', 'n/a']].forEach((r) => t2.tBodies[0].appendChild(el('tr', {}, r.map((v) => el('td', {text: v})))));
const box = el('div', {}, [el('div', {class: 'scroll'}, [t2])]); doc.body.appendChild(box);
Promise.resolve().then(() => {
  assert.deepStrictEqual(names(t2), ['y', 'x', 'z'], 'queue asc after the microtask');
  assert.ok(box.children[0].classList.contains('sort-tools'));
  // Groups and pins.
  const t3 = el('table', {'data-sort': 'g'}, [el('thead', {}, [el('tr', {}, [el('th', {text: 'N'}), el('th', {text: 'V'})])]), el('tbody', {}, [
    el('tr', {'data-sort-pin': 'top'}, [el('td', {text: 'zero', colspan: 2})]),
    el('tr', {}, [el('td', {text: 'r1'}), el('td', {text: '2'})]),
    el('tr', {'data-sort-follow': ''}, [el('td', {text: 'r1-detail'})]),
    el('tr', {}, [el('td', {text: 'r2'}), el('td', {text: '1'})]),
    el('tr', {'data-sort-pin': 'bottom'}, [el('td', {text: 'total'}), el('td', {text: '0'})]),
  ])]);
  doc.body.appendChild(t3); S.scan(doc);
  t3.tHead.rows[0].cells[1].dispatch('click', {target: t3.tHead.rows[0].cells[1]});
  assert.deepStrictEqual(names(t3), ['zero', 'r2', 'r1', 'r1-detail', 'total']);
  // Two header rows with a group.
  const t4 = el('table', {}, [el('thead', {}, [
    el('tr', {}, [Object.assign(el('th', {text: 'Model'}), {rowSpan: 2}), Object.assign(el('th', {text: 'Finds'}), {colSpan: 2}), Object.assign(el('th', {text: 'Speed'}), {rowSpan: 2})]),
    el('tr', {}, [el('th', {text: 'All', 'data-sort-col': 'finds:all'}), el('th', {text: 'Day'})]),
  ]), el('tbody', {}, [
    el('tr', {}, [el('td', {text: 'm1'}), el('td', {text: '50%'}), el('td', {text: '10%'}), el('td', {text: '30 ms'})]),
    el('tr', {class: 'lineup-out'}, [el('td', {text: 'm0'}), el('td', {text: '99%'}), el('td', {text: '99%'}), el('td', {text: '1 ms'})]),
    el('tr', {}, [el('td', {text: 'm2'}), el('td', {text: '70%'}), el('td', {text: '5%'}), el('td', {text: '1.2 s'})]),
  ])]);
  doc.body.appendChild(t4); S.attach(t4, {key: 'lineup'});
  const leafs = t4._prismSort.cols.map((c) => [c.col, c.label]);
  assert.deepStrictEqual(JSON.parse(JSON.stringify(leafs)), [['0', 'Model'], ['finds:all', 'Finds · All'], ['2', 'Finds · Day'], ['3', 'Speed']]);
  const dayTh = t4.tHead.rows[1].cells[1];
  dayTh.dispatch('click', {target: dayTh});
  assert.deepStrictEqual(names(t4), ['m2', 'm1', 'm0'], 'day asc, the out row last');
  const sp = t4.tHead.rows[0].cells[2];
  sp.dispatch('click', {target: sp});
  assert.deepStrictEqual(names(t4), ['m1', 'm2', 'm0'], 'speed: 30 ms < 1.2 s');
  // state / restore for an unkeyed table.
  const t6 = el('table', {}, [el('thead', {}, [el('tr', {}, [el('th', {text: 'N'})])]), el('tbody', {}, [el('tr', {}, [el('td', {text: 'b'})]), el('tr', {}, [el('td', {text: 'a'})])])]);
  doc.body.appendChild(t6); S.attach(t6, {});
  assert.strictEqual(S.state(t6), null);
  S.restore(t6, [{col: '0', dir: 'asc'}]);
  assert.deepStrictEqual(names(t6), ['a', 'b']);
  assert.deepStrictEqual(JSON.parse(JSON.stringify(S.state(t6))), [{col: '0', dir: 'asc'}]);
  // Best-first default label.
  const t7 = el('table', {'data-sort': 'best', 'data-sort-default': 'best'}, [el('thead', {}, [el('tr', {}, [el('th', {text: 'N'})])]), el('tbody', {}, [el('tr', {'data-best': '2'}, [el('td', {text: 'b'})]), el('tr', {}, [el('td', {text: 'c'})]), el('tr', {'data-best': '1'}, [el('td', {text: 'a'})])])]);
  doc.body.appendChild(t7); S.scan(doc);
  assert.deepStrictEqual(names(t7), ['a', 'b', 'c']);
  assert.strictEqual(t7._prismSort.reset.textContent, 'Best first');
  // Sections: a click sorts inside each `data-sort-group` section, Best first puts them back.
  const t8 = el('table', {'data-sort': 'sections', 'data-sort-default': 'best'}, [el('thead', {}, [el('tr', {}, [el('th', {text: 'N'}), el('th', {text: 'R'})])]), el('tbody', {}, [
    el('tr', {'data-sort-group': ''}, [el('td', {text: 'set A'})]),
    el('tr', {'data-best': '2'}, [el('td', {text: 'a1'}), el('td', {text: '0.9'})]),
    el('tr', {'data-sort-follow': ''}, [el('td', {text: 'a1-detail'})]),
    el('tr', {'data-best': '1'}, [el('td', {text: 'a2'}), el('td', {text: '0.5'})]),
    el('tr', {'data-sort-group': ''}, [el('td', {text: 'set B'})]),
    el('tr', {'data-best': '1'}, [el('td', {text: 'b1'}), el('td', {text: '0.1'})]),
    el('tr', {'data-best': '2'}, [el('td', {text: 'b2'}), el('td', {text: '0.95'})]),
  ])]);
  doc.body.appendChild(t8); S.scan(doc);
  assert.deepStrictEqual(names(t8), ['set A', 'a2', 'a1', 'a1-detail', 'set B', 'b1', 'b2'], 'best first in each section');
  const rTh = t8.tHead.rows[0].cells[1];
  rTh.dispatch('click', {target: rTh});
  assert.deepStrictEqual(names(t8), ['set A', 'a2', 'a1', 'a1-detail', 'set B', 'b1', 'b2'], 'R ascending in each section');
  rTh.dispatch('click', {target: rTh});
  assert.deepStrictEqual(names(t8), ['set A', 'a1', 'a1-detail', 'a2', 'set B', 'b2', 'b1'], 'R descending, the sections stay');
  t8._prismSort.reset.dispatch('click');
  assert.deepStrictEqual(names(t8), ['set A', 'a2', 'a1', 'a1-detail', 'set B', 'b1', 'b2']);
  // The sort button goes in a nowrap span with the header's last word, or with its last element
  // (a glossary button): a wrapping header never leaves it alone on a line. The label is the same.
  const gTerm = el('button', {class: 'term', text: 'Recall'});
  const t9 = el('table', {'data-sort': 'glue'}, [el('thead', {}, [el('tr', {}, [el('th', {text: 'False alerts /100 empty '}), el('th', {}, [gTerm]), el('th', {text: 'p95'})])]),
    el('tbody', {}, [el('tr', {}, [el('td', {text: '1'}), el('td', {text: '2'}), el('td', {text: '3'})])])]);
  doc.body.appendChild(t9); S.scan(doc);
  const [g0, g1, g2] = t9.tHead.rows[0].cells;
  const glueOf = (th) => th.querySelector('button.th-sort').parentNode;
  assert.strictEqual(glueOf(g0).className, 'th-sort-glue');
  assert.strictEqual(glueOf(g0).firstChild.nodeValue, 'empty', 'the last word');
  assert.strictEqual(g0.firstChild.nodeValue, 'False alerts /100 ', 'the rest stays before it');
  assert.strictEqual(g0.querySelector('button.th-sort').getAttribute('aria-label'), 'Sort by False alerts /100 empty');
  assert.strictEqual(glueOf(g1).firstChild, gTerm, 'a glossary button goes in with it');
  assert.strictEqual(g1.querySelector('button.th-sort').getAttribute('aria-label'), 'Sort by Recall');
  assert.strictEqual(glueOf(g2).firstChild.nodeValue, 'p95');
  assert.strictEqual(g2.childNodes.length, 1, 'a one-word header: only the span');
  console.log('dom ok');
}).catch((e) => { console.error(e); process.exit(1); });
"#;

#[test]
fn the_page_part_sorts_a_table_and_keeps_its_state() {
    if have_node() {
        node(PAGE);
    }
}
