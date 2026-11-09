//! Phones: the screen rules of `assets/live.js` (`PrismViewport`: whether an on-screen keyboard
//! is open, so the sticky Save / decision bars stop sticking, and how far the bottom restart
//! banner rises over those bars) and `assets/browse.js` focusing the filled field only under a
//! mouse or pen (on a phone the focus would open the keyboard), run in `node` with a stub DOM.
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
    let dir = std::env::temp_dir().join(format!("bop-js-viewport-{}", uuid::Uuid::new_v4()));
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

const KEYBOARD: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({console, Promise});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);   // no `document`: the pure part only
const V = vm.runInContext('PrismViewport', ctx);
const open = (vv, inner, tag, type, scale) => V.keyboardOpen(vv, inner, tag, type, scale);
// iPhone portrait: innerHeight stays 664, the visual viewport drops to ~330 under the keyboard.
assert.strictEqual(open(330, 664, 'INPUT', 'number'), true, 'a number field with the keyboard up');
assert.strictEqual(open(330, 664, 'input', 'text'), true);
assert.strictEqual(open(330, 664, 'INPUT', ''), true, 'no type is text');
assert.strictEqual(open(330, 664, 'INPUT', undefined), true);
assert.strictEqual(open(330, 664, 'TEXTAREA', 'textarea'), true);
assert.strictEqual(open(330, 664, 'SELECT', 'select-one'), true);
// No keyboard: the toolbar collapsing or appearing is far less than a quarter.
assert.strictEqual(open(600, 664, 'INPUT', 'text'), false);
assert.strictEqual(open(664 * 0.75, 664, 'INPUT', 'text'), false, 'exactly three quarters is no keyboard');
assert.strictEqual(open(664 * 0.75 - 1, 664, 'INPUT', 'text'), true);
// Not a typing field: whatever the height, the bars stay.
for (const [tag, type] of [['INPUT', 'checkbox'], ['INPUT', 'radio'], ['INPUT', 'range'], ['INPUT', 'button'],
    ['INPUT', 'submit'], ['INPUT', 'reset'], ['INPUT', 'file'], ['INPUT', 'color'], ['BUTTON', 'submit'],
    ['BODY', undefined], [undefined, undefined], [null, null], ['A', '']]) {
  assert.strictEqual(open(300, 664, tag, type), false, tag + ' ' + type);
}
// Pinch zoom shrinks the visual viewport in CSS px with no keyboard: the scale undoes it.
assert.strictEqual(open(664 / 2, 664, 'INPUT', 'text', 2), false, 'zoomed in 2x, no keyboard');
assert.strictEqual(open(330 / 2, 664, 'INPUT', 'text', 2), true, 'zoomed in 2x, keyboard up');
assert.strictEqual(open(330, 664, 'INPUT', 'text', 0), true, 'a bad scale counts as 1');
assert.strictEqual(open(330, 664, 'INPUT', 'text', NaN), true);
// Nonsense sizes: no keyboard.
assert.strictEqual(open(0, 664, 'INPUT', 'text'), false);
assert.strictEqual(open(330, 0, 'INPUT', 'text'), false);
assert.strictEqual(open(NaN, 664, 'INPUT', 'text'), false);

// The banner at the bottom rises over the bars stuck to the bottom of an 800 px window.
const lift = V.bottomLift;
assert.strictEqual(lift([], 800), 0);
assert.strictEqual(lift(null, 800), 0);
assert.strictEqual(lift([{top: 742, bottom: 800}], 800), 58, 'a bar stuck at the bottom');
assert.strictEqual(lift([{top: 742.4, bottom: 799.5}], 800), 58, 'rounded up, a fraction off the edge still stuck');
assert.strictEqual(lift([{top: 400, bottom: 458}], 800), 0, 'a bar in the middle of the page (its form ended)');
assert.strictEqual(lift([{top: 820, bottom: 878}], 800), 0, 'below the window');
assert.strictEqual(lift([{top: 0, bottom: 0}], 800), 0, 'not laid out');
assert.strictEqual(lift([{top: 742, bottom: 800}, {top: 700, bottom: 800}], 800), 100, 'the tallest one');
console.log('ok');
"#;

/// browse.js with the native picker answering a path: the field gets the path either way, the
/// focus only when the primary pointer is fine.
const BROWSE_FOCUS: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const tick = () => new Promise(r => setImmediate(r));

function node(tag) {
  const n = {tagName: tag.toUpperCase(), children: [], dataset: {}, hidden: false, textContent: '', listeners: {},
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    insertBefore(c) { this.children.push(c); c.parentNode = this; return c; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    setAttribute() {},
    closest() { return null; }};
  return n;
}
async function pickWith(pointer) {
  const focused = [], events = [];
  const input = node('input');
  input.value = '';
  input.focus = () => focused.push('input');
  input.dispatchEvent = (e) => events.push(e.type);
  const form = {elements: {path: input}};
  const button = node('button');
  button.dataset = {browse: 'model', target: 'path'};
  button.form = form;
  button.closest = (sel) => sel.indexOf('button.browse') === 0 ? button : null;
  const wrap = node('span'); wrap.appendChild(button);
  const docListeners = {};
  const document = {
    getElementById: () => null,          // no in-page dialog: native only
    querySelector: (sel) => sel.indexOf('button.browse') === 0 ? button : null,
    querySelectorAll: (sel) => sel.indexOf('button.browse') === 0 ? [button] : [],
    createElement: (tag) => node(tag),
    addEventListener: (t, f) => { docListeners[t] = f; },
    body: node('body')
  };
  const sb = {console, Promise, document, Event: class { constructor(t) { this.type = t; } }, setTimeout, clearTimeout,
    AbortController,
    fetch: (url, init) => Promise.resolve({status: 200, ok: true,
      json: () => Promise.resolve(init && init.method === 'POST' ? {path: '/models/a.onnx'} : {available: true})})};
  if (pointer !== null) sb.matchMedia = (q) => ({matches: q === '(pointer: fine)' ? pointer === 'fine' : false, media: q});
  sb.window = sb;
  vm.runInContext(src, vm.createContext(sb));
  await tick(); await tick();
  assert.strictEqual(button.hidden, false, 'the native picker shows the button');
  docListeners.click({target: button});
  for (let i = 0; i < 5; i++) await tick();
  assert.strictEqual(input.value, '/models/a.onnx', pointer + ': the path is filled');
  assert.deepStrictEqual(events, ['input'], pointer + ': the page hears the change');
  return focused.length > 0;
}
(async () => {
  assert.strictEqual(await pickWith('fine'), true, 'a mouse: the field takes the focus');
  assert.strictEqual(await pickWith('coarse'), false, 'a finger: no focus, so no keyboard');
  assert.strictEqual(await pickWith(null), true, 'no matchMedia: as before');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

#[test]
fn the_keyboard_rule_and_the_banner_lift() {
    if have_node() {
        run(KEYBOARD, "live.js");
    }
}

#[test]
fn a_picked_path_focuses_its_field_only_under_a_fine_pointer() {
    if have_node() {
        run(BROWSE_FOCUS, "browse.js");
    }
}
