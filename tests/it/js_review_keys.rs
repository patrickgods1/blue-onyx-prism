//! The Review tab's keys (`assets/review-keys.js`): the one list behind the table under Panel
//! options and the `?` overlay, the two drawn from it on a fake page, the overlay's wiring, and
//! which fields take a key themselves (`typing`).
//!
//! [`DRIVER`] was first run against the keys section of `assets/review.js` before it moved to its
//! own file (characterisation).
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI), which
//! makes a missing `node` a failure: a skip must not read as a pass there.
#![allow(clippy::unwrap_used)]

use crate::js_harness::{asset, have_node, run_node_ok};

/// `process.argv[2]`: the module.
const DRIVER: &str = r#"
const assert = require('assert');
// A fake page: elements keep their children, setting the text drops them.
const node = (tag) => ({
  tag, children: [], className: '', id: '', text: '', listeners: {},
  appendChild(c) { this.children.push(c); },
  get textContent() { return this.text; }, set textContent(t) { this.text = t; this.children = []; },
  addEventListener(type, f) { (this.listeners[type] = this.listeners[type] || []).push(f); },
  fire(type, ev) { (this.listeners[type] || []).forEach((f) => f(ev || {})); }
});
const el = (tag, cls, text) => { const e = node(tag); if (cls) e.className = cls; if (text != null) e.text = text; return e; };
const button = (text, fn, cls) => ({ tag: 'button', text, fn, cls });
let pageBody = node('tbody');
globalThis.document = {
  querySelector(q) { assert.strictEqual(q, '.review-keys tbody'); return pageBody; },
  createTextNode(t) { return { tag: '#text', text: t }; }
};
require(process.argv[2]);
const K = globalThis.ReviewKeys;

// The list: three groups in order, one class row, one note.
const groups = K.KEYS.map((k) => k[0]);
assert.deepStrictEqual([...new Set(groups)], ['Decide', 'Navigate', 'View']);
assert.deepStrictEqual(groups.slice().sort((a, b) => ['Decide', 'Navigate', 'View'].indexOf(a) - ['Decide', 'Navigate', 'View'].indexOf(b)), groups, 'grouped');
const count = (g) => groups.filter((x) => x === g).length;
assert.deepStrictEqual([count('Decide'), count('Navigate'), count('View')], [21, 10, 10]);
assert.deepStrictEqual(K.KEYS.filter((k) => k[2] == null).map((k) => k[1]), ['1-9, 0']);
assert.deepStrictEqual(K.KEYS.filter((k) => !k[1]).length, 1);
assert.ok(K.KEYS.every((k) => k.length === 3));
assert.strictEqual(K.KEYS[0][1], 'a');
assert.strictEqual(K.KEYS[K.KEYS.length - 1][1], '?');

// The page's table and the overlay.
let modal = 0, focus = [];
const dialog = node('dialog');
dialog.open = false;
dialog.showModal = function () { modal++; this.open = true; };
dialog.close = function () { this.open = false; this.fire('close'); };
const canvas = { focus(o) { focus.push(o); } };
const keysBtn = node('button');
const t = K.init({ el, button, dialog, canvas, keysBtn });

// Table: a group row (th over two columns), then its keys.
const rows = pageBody.children;
assert.strictEqual(rows.length, 3 + K.KEYS.length);
assert.strictEqual(rows[0].className, 'rv-keys-group');
assert.strictEqual(rows[0].children[0].tag, 'th');
assert.strictEqual(rows[0].children[0].text, 'Decide');
assert.strictEqual(rows[0].children[0].colSpan, 2);
assert.deepStrictEqual(rows.filter((r) => r.className === 'rv-keys-group').map((r) => r.children[0].text), ['Decide', 'Navigate', 'View']);
assert.strictEqual(rows[22].children[0].text, 'Navigate', 'after the 21 Decide keys');
assert.deepStrictEqual([rows[1].children[0].text, rows[1].children[1].text], ['a', 'accept the selected box (keep a drawn one) and go to the next']);
// The class row: text, the span review.js fills (with its id on the page's table), text.
const classRow = rows.find((r) => r.children[0] && r.children[0].text === '1-9, 0');
const cells = classRow.children[1].children;
assert.deepStrictEqual(cells.map((c) => c.tag), ['#text', 'span', '#text']);
assert.strictEqual(cells[0].text, 'class 1-10');
assert.strictEqual(cells[1].className, 'rv-key-classes');
assert.strictEqual(cells[1].id, 'rv-key-classes');
assert.strictEqual(cells[2].text, ": relabel the selected box (accepting it) and go to the next, else the class of new boxes; the bar's It's a… lists them all");
// The note: no key, muted.
const note = rows.find((r) => r.children[0] && r.children[0].text === '' && r.children.length === 2);
assert.strictEqual(note.children[1].className, 'muted');
assert.ok(note.children[1].text.indexOf('The next box is the next likely') === 0);

// Overlay: a head (title, Close), then a column per group, each a table of its keys.
const [head, cols] = dialog.children;
assert.strictEqual(dialog.children.length, 2);
assert.strictEqual(head.className, 'rv-keys-head');
assert.deepStrictEqual(head.children.map((c) => [c.tag, c.text]), [['h3', 'Keyboard'], ['button', 'Close']]);
assert.strictEqual(head.children[1].cls, 'secondary');
assert.strictEqual(cols.className, 'rv-keys-cols');
assert.deepStrictEqual(cols.children.map((s) => [s.tag, s.children[0].text, s.children[1].className, s.children[1].children[0].children.length]),
  [['section', 'Decide', 'review-keys', 21], ['section', 'Navigate', 'review-keys', 10], ['section', 'View', 'review-keys', 10]]);
const overlaySpan = cols.children[0].children[1].children[0].children.find((r) => r.children[0].text === '1-9, 0').children[1].children[1];
assert.strictEqual(overlaySpan.className, 'rv-key-classes');
assert.strictEqual(overlaySpan.id, '', 'one id on the page');

// Opening: once (open again while open does nothing), from `?` or the Keys button.
t.open();
t.open();
assert.strictEqual(modal, 1);
// A click inside leaves it open; one on the backdrop (the dialog itself) closes it, and the frame
// takes the focus back without scrolling.
dialog.fire('click', { target: head });
assert.strictEqual(dialog.open, true);
dialog.fire('click', { target: dialog });
assert.strictEqual(dialog.open, false);
assert.deepStrictEqual(focus, [{ preventScroll: true }]);
keysBtn.fire('click');
assert.strictEqual(modal, 2);
head.children[1].fn();
assert.strictEqual(dialog.open, false, 'Close closes');
// Without showModal (an old browser): the open attribute.
const bare = node('dialog');
bare.open = false;
bare.setAttribute = function (k, v) { this.attr = [k, v]; };
pageBody = null;
K.init({ el, button, dialog: bare, canvas, keysBtn: null }).open();
assert.deepStrictEqual(bare.attr, ['open', '']);
assert.strictEqual(bare.children.length, 2, 'the overlay without a page table');

// typing: text fields take every key, a checkbox none, a box's class select its navigation keys.
const field = (tagName, type, extra) => Object.assign({ tagName, type, classList: { contains: (c) => !!(extra && extra.cls === c) } }, extra || {});
assert.strictEqual(K.typing(field('INPUT', 'text'), 'a'), true);
assert.strictEqual(K.typing(field('INPUT', 'number'), 'a'), true);
for (const ty of ['checkbox', 'radio', 'button', 'submit', 'reset']) assert.strictEqual(K.typing(field('INPUT', ty), 'a'), false, ty);
assert.strictEqual(K.typing(field('TEXTAREA'), 'a'), true);
assert.strictEqual(K.typing(field('SELECT'), 'a'), true);
const boxSelect = field('SELECT', undefined, { cls: 'bx-label' });
assert.strictEqual(K.typing(boxSelect, 'a'), false);
assert.strictEqual(K.typing(boxSelect, '4'), false);
for (const k of ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter', ' ', 'Home', 'End', 'PageUp', 'PageDown', 'Tab']) assert.strictEqual(K.typing(boxSelect, k), true, k);
assert.strictEqual(K.typing(boxSelect, 'Escape'), false);
assert.strictEqual(K.typing(field('DIV', undefined, { isContentEditable: true }), 'a'), true);
assert.strictEqual(K.typing(field('DIV'), 'a'), false);
assert.strictEqual(K.typing(field('BUTTON'), 'Enter'), false);
assert.strictEqual(K.typing(null, 'a'), false);
console.log('ok');
"#;

#[test]
fn one_key_list_fills_the_table_and_the_overlay() {
    if !have_node() {
        return;
    }
    run_node_ok(DRIVER, &[asset("review-keys.js").as_os_str()]);
}
