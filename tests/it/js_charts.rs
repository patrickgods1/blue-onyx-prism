//! Charts and hover on a touch screen: `assets/bench-charts.js` (the line charts' readout on a
//! tap, kept until a tap elsewhere or Esc, a sideways finger drag that scrubs it, an up-down one
//! that leaves nothing behind, the ROC dots above the hit rect with their text on a tap, the
//! narrower viewBox of a phone, the scatter's bigger rings for a finger) and `assets/live.js`
//! `PrismPointer` (a mouse over a chart defers a re-render of its section, a finger never does: iOS
//! keeps `:hover` set after a tap, which froze the section), run in `node` with a stub DOM.
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

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(name)
}

/// Run `driver_js` with `asset` as its argument; it asserts and exits non-zero on a failure.
fn run(driver_js: &str, asset_name: &str) {
    let dir = std::env::temp_dir().join(format!("bop-js-charts-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let driver = dir.join("driver.js");
    std::fs::write(&driver, driver_js).unwrap();
    let out = Command::new("node")
        .arg(&driver)
        .arg(asset(asset_name))
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

/// A stub DOM just big enough for bench-charts.js, loaded at window width `vw` with a coarse or
/// fine pointer.
const STUB: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
class N {
  constructor(tag) { this.tagName = tag; this.attrs = {}; this.children = []; this.parentNode = null; this.listeners = {}; this._text = ''; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this.children.length ? this.children.map(c => c.textContent).filter(Boolean).join('\n') : this._text; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  hasAttribute(k) { return k in this.attrs; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  replaceChildren(...cs) { this.children = []; cs.forEach(c => this.appendChild(c)); }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t, e) { (this.listeners[t] || []).forEach(f => f(Object.assign({target: this}, e))); }
  contains(n) { for (; n; n = n.parentNode) if (n === this) return true; return false; }
  getBoundingClientRect() { const vb = (this.attrs.viewBox || '0 0 0 0').split(' ').map(Number); return {left: 0, top: 0, width: vb[2], height: vb[3]}; }
  byClass(c) { return this.children.find(k => (k.attrs.class || '').split(' ').indexOf(c) >= 0); }
}
function load(vw, coarse) {
  const docListeners = {};
  const document = {
    documentElement: {clientWidth: vw},
    createElementNS: (ns, tag) => new N(tag),
    createElement: (tag) => new N(tag),
    addEventListener: (t, f) => { (docListeners[t] = docListeners[t] || []).push(f); },
  };
  const has = v => v !== null && v !== undefined && !(typeof v === 'number' && isNaN(v));
  const window = {
    innerWidth: vw,
    matchMedia: q => ({matches: q === '(pointer: coarse)' ? coarse : false}),
    PrismBench: {el: (t, c, x) => { const e = new N(t); if (c) e.className = c; if (x !== undefined) e.textContent = x; return e; }, has, fmt: {thr: v => String(v), ms: v => v + ' ms', pct: v => v + '%'}},
  };
  const ctx = vm.createContext({console, document, window, setTimeout});
  vm.runInContext(src, ctx);
  const doc = (t, e) => (docListeners[t] || []).forEach(f => f(e));
  return {C: window.PrismBench.charts, doc};
}
// A line chart over x 0..1 with one series; returns its parts.
function chart(C, extra) {
  const pts = [0, 0.25, 0.5, 0.75, 1].map(x => ({x, y: 1 - x}));
  const svg = C.lineChart([{id: 'a', label: 'A', points: pts}], Object.assign({title: 't', xName: 'x'}, extra || {}));
  return {svg, hit: svg.byClass('thr-hit'), tip: svg.byClass('thr-tip'), guide: svg.byClass('thr-guide')};
}
const shown = c => c.tip.attrs.visibility === 'visible';
const pinned = c => c.svg.hasAttribute('data-pinned');
const finger = (x, y) => ({pointerType: 'touch', clientX: x, clientY: y, buttons: 1});
const mouse = (x, y) => ({pointerType: 'mouse', clientX: x, clientY: y, buttons: 0});
"#;

const TAP: &str = r#"
{ // A tap shows the readout and keeps it: through the pointerleave after the finger lifts, a tap
  // inside the same chart, until a tap elsewhere.
  const {C, doc} = load(1200, false);
  const c = chart(C);
  assert.strictEqual(c.svg.attrs.viewBox, '0 0 560 290', 'a wide window keeps the asked size');
  assert.ok(!shown(c));
  c.hit.fire('pointerdown', finger(300, 100));
  assert.ok(!shown(c), 'nothing yet: it may be the start of a scroll');
  c.hit.fire('pointerup', finger(300, 100));
  assert.ok(shown(c) && pinned(c), 'the readout after the tap');
  assert.ok(c.tip.textContent.indexOf('A: ') >= 0, c.tip.textContent);
  c.hit.fire('pointerleave', {pointerType: 'touch'});
  assert.ok(shown(c), 'the pointerleave after a finger lifts keeps it');
  doc('pointerdown', {target: c.hit});
  assert.ok(shown(c), 'a tap inside the chart keeps it');
  doc('pointerdown', {target: new N('div')});
  assert.ok(!shown(c) && !pinned(c), 'a tap elsewhere clears it');
  assert.strictEqual(c.guide.attrs.visibility, 'hidden');
}
{ // Esc clears it; so does a tap on another chart, which pins its own.
  const {C, doc} = load(1200, false);
  const a = chart(C), b = chart(C);
  a.hit.fire('pointerdown', finger(200, 100)); a.hit.fire('pointerup', finger(200, 100));
  doc('keydown', {key: 'Enter'});
  assert.ok(shown(a), 'another key keeps it');
  doc('keydown', {key: 'Escape'});
  assert.ok(!shown(a) && !pinned(a), 'Esc clears it');
  a.hit.fire('pointerdown', finger(200, 100)); a.hit.fire('pointerup', finger(200, 100));
  doc('pointerdown', {target: b.hit});
  b.hit.fire('pointerdown', finger(200, 100)); b.hit.fire('pointerup', finger(200, 100));
  assert.ok(!shown(a) && !pinned(a) && shown(b) && pinned(b), 'one pinned readout at a time');
}
{ // A mouse: the readout follows it and goes when it leaves (as before), never pinned.
  const {C} = load(1200, false);
  const c = chart(C);
  c.hit.fire('pointermove', mouse(300, 100));
  assert.ok(shown(c) && !pinned(c));
  const x1 = c.guide.attrs.x1;
  c.hit.fire('pointermove', mouse(500, 100));
  assert.notStrictEqual(c.guide.attrs.x1, x1, 'the guide follows the mouse');
  c.hit.fire('pointerup', mouse(500, 100));
  assert.ok(!pinned(c), 'a click pins nothing');
  c.hit.fire('pointerleave', {pointerType: 'mouse'});
  assert.ok(!shown(c));
}
{ // A finger dragging up or down scrolls the page: no readout left behind.
  const {C} = load(1200, false);
  const c = chart(C);
  c.hit.fire('pointerdown', finger(300, 100));
  c.hit.fire('pointermove', finger(302, 120));
  assert.ok(!shown(c), 'an up-down move shows nothing');
  c.hit.fire('pointercancel', {pointerType: 'touch'});
  c.hit.fire('pointerleave', {pointerType: 'touch'});
  assert.ok(!shown(c) && !pinned(c));
}
{ // A sideways drag scrubs: the readout follows the finger and stays after the page takes over.
  const {C} = load(1200, false);
  const c = chart(C);
  c.hit.fire('pointerdown', finger(100, 100));
  c.hit.fire('pointermove', finger(104, 100));
  assert.ok(!shown(c), 'a few pixels are not a drag yet');
  c.hit.fire('pointermove', finger(140, 102));
  assert.ok(shown(c) && pinned(c), 'scrubbing');
  const x1 = c.guide.attrs.x1;
  c.hit.fire('pointermove', finger(480, 110));
  assert.notStrictEqual(c.guide.attrs.x1, x1, 'the guide follows the finger');
  c.hit.fire('pointercancel', {pointerType: 'touch'});
  assert.ok(shown(c) && pinned(c), 'a cancelled scrub keeps its last readout');
}
console.log('ok');
"#;

const DOTS_AND_SIZES: &str = r#"
// The viewBox for the width a chart shows at: as asked when that is wider, else that width.
{
  const {C} = load(1200, false);
  const f = (w, h, avail) => Array.from(C.fitSize(w, h, avail));   // an array of the context, not this realm
  assert.deepStrictEqual(f(560, 290, 1200), [560, 290]);
  assert.deepStrictEqual(f(560, 290, 560), [560, 290]);
  assert.deepStrictEqual(f(560, 290, 0), [560, 290], 'no width known: as asked');
  assert.deepStrictEqual(f(560, 290, NaN), [560, 290]);
  assert.deepStrictEqual(f(560, 290, 350), [350, 209], 'as wide as it shows, a bit taller than to scale');
  assert.deepStrictEqual(f(620, 340, 350), [350, 245]);
  assert.deepStrictEqual(f(330, 300, 290), [290, 264], 'the ROC in a phone card');
  assert.deepStrictEqual(f(320, 210, 311), [311, 204], 'the training chart in a phone card');
  assert.deepStrictEqual(f(330, 300, 340), [330, 300], 'never wider than asked');
  assert.deepStrictEqual(f(560, 290, 200), [240, 209], 'at least 240');
  assert.deepStrictEqual(f(560, 290, 333.6), [334, 209]);
  // Shown at the width it was drawn for, the 11 px ticks are 11 px.
  for (const avail of [240, 290, 334, 390, 559]) {
    const [w] = f(560, 290, avail);
    assert.ok(Math.abs(avail / w * 11 - 11) < 0.05, avail + ': ' + w);
  }
}
{ // A wide window draws as asked (the desktop look unchanged); a phone's window less the padding.
  assert.strictEqual(chart(load(1200, false).C).svg.attrs.viewBox, '0 0 560 290');
  assert.strictEqual(chart(load(600, false).C).svg.attrs.viewBox, '0 0 560 290');
  assert.strictEqual(chart(load(599, false).C).svg.attrs.viewBox, '0 0 499 258');
}
{ // On a phone the chart draws the narrower viewBox.
  const {C} = load(390, true);
  assert.strictEqual(chart(C).svg.attrs.viewBox, '0 0 290 209');
}
{ // ROC dots: painted after the hit rect (their <title> gets the mouse), the readout on top of
  // them; a tap on one shows its text and keeps it; a mouse gets the title instead.
  const {C} = load(1200, false);
  const c = chart(C, {hover: 'xy', dots: [{x: 0.5, y: 0.5, cls: 'roc-dot-best', label: 'best 0.4: alert rate 80%', lines: ['best 0.4', 'alert rate 80%']}]});
  const kids = c.svg.children, dot = c.svg.byClass('roc-dot-best');
  assert.ok(kids.indexOf(c.hit) < kids.indexOf(dot), 'the dot is above the hit rect');
  assert.ok(kids.indexOf(dot) < kids.indexOf(c.tip), 'the readout is above the dot');
  assert.strictEqual(dot.children[0].tagName, 'title');
  dot.fire('pointerup', mouse(280, 140));
  assert.ok(!shown(c), 'a mouse click on a dot: its title, no readout');
  dot.fire('pointerup', finger(280, 140));
  assert.ok(shown(c) && pinned(c));
  assert.strictEqual(c.tip.textContent, 'best 0.4\nalert rate 80%');
}
{ // The scatter's hit rings: 11 for a mouse, 22 for a finger.
  for (const [coarse, r] of [[false, '11'], [true, '22']]) {
    const {C} = load(1200, coarse);
    const svg = C.scatter([{key: 'k', x: 10, y: 0.5, label: 'm'}], {});
    const g = svg.byClass('pb-pt'), ring = g.byClass('pb-pt-ring');
    assert.strictEqual(ring.attrs.r, r, 'coarse ' + coarse);
  }
}
console.log('ok');
"#;

/// `PrismPointer` of live.js, no `document`: a mouse or a pen entering a chart or a
/// `.keep-hover` element marks it, leaving clears it; a finger never marks anything; `within`
/// finds a mark or a pinned chart under a section.
const POINTER: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({console, Promise});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);   // no `document`: the pure part only
const P = vm.runInContext('PrismPointer', ctx);
function el(tag, cls) {
  return {tagName: tag, cls: cls || '', attrs: {}, kids: [],
    matches(sel) { assert.strictEqual(sel, 'svg, .keep-hover'); return this.tagName === 'svg' || this.cls.split(' ').indexOf('keep-hover') >= 0; },
    setAttribute(k, v) { this.attrs[k] = v; }, removeAttribute(k) { delete this.attrs[k]; },
    querySelector(sel) {
      assert.strictEqual(sel, '[data-pointer-in], svg[data-pinned]');
      const walk = n => { for (const k of n.kids) { if ('data-pointer-in' in k.attrs || (k.tagName === 'svg' && 'data-pinned' in k.attrs)) return k; const f = walk(k); if (f) return f; } return null; };
      return walk(this);
    }};
}
const section = el('div'), svg = el('svg'), term = el('button', 'term keep-hover'), plain = el('div'), rect = el('rect');
section.kids = [svg, term, plain]; svg.kids = [rect];
assert.ok(P.hovers('mouse') && P.hovers('pen') && !P.hovers('touch') && !P.hovers(undefined));
assert.ok(!P.within(section));
P.note(svg, 'touch', true);
assert.ok(!P.within(section), 'a finger on a chart is no hover: the section re-renders');
P.note(svg, 'mouse', true);
assert.ok(P.within(section), 'a mouse over a chart holds the re-render');
P.note(rect, 'mouse', true); P.note(plain, 'mouse', true);
assert.ok(!('data-pointer-in' in rect.attrs) && !('data-pointer-in' in plain.attrs), 'only charts and .keep-hover are marked');
P.note(svg, 'mouse', false);
assert.ok(!P.within(section), 'the mouse left');
P.note(term, 'pen', true);
assert.ok(P.within(section), 'a hovering pen over a term');
P.note(term, 'touch', false);
assert.ok(P.within(section), 'a finger does not clear a mouse or pen mark');
P.note(term, 'pen', false);
assert.ok(!P.within(section));
svg.attrs['data-pinned'] = '';
assert.ok(P.within(section), 'a readout a finger pinned holds the re-render too');
assert.ok(!P.within(null) && !P.within({}));
// install(): capture listeners for both events, which pass the event's target and type on.
const seen = {};
P.install({addEventListener: (t, f, capture) => { assert.strictEqual(capture, true, t); seen[t] = f; }});
delete svg.attrs['data-pinned'];
seen.pointerenter({target: svg, pointerType: 'mouse'});
assert.ok(P.within(section));
seen.pointerleave({target: svg, pointerType: 'mouse'});
assert.ok(!P.within(section));
console.log('ok');
"#;

#[test]
fn a_tap_shows_the_chart_readout_and_keeps_it() {
    if !have_node() {
        return;
    }
    run(&format!("{STUB}{TAP}"), "bench-charts.js");
}

#[test]
fn roc_dots_take_the_pointer_and_phones_get_readable_charts() {
    if !have_node() {
        return;
    }
    run(&format!("{STUB}{DOTS_AND_SIZES}"), "bench-charts.js");
}

#[test]
fn a_mouse_over_a_chart_defers_re_renders_and_a_finger_does_not() {
    if !have_node() {
        return;
    }
    run(POINTER, "live.js");
}
