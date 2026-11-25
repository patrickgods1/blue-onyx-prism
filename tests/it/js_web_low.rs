//! Small fixes of the web assets (audit 2026-11, phase 8), run in `node` with fake timers, a fake
//! `fetch` and stub DOMs:
//! - QA-113: the restart page and the restart button wait for another generation or another
//!   process (`epoch`), and a restart pressed before the first `/v1/config` answer reads the
//!   current generation first (`assets/live.js`).
//! - QA-115: a Jobs list answer that is not the list backs the poll off and says so
//!   (`assets/train-area.js`).
//! - QA-117: a class name counts code points like the server (`assets/class-picker.js`).
//! - Audit extras 37 / 38: a proxy's HTML answer reads "HTTP 502" on the Import preview, not
//!   the parser's message (`assets/train-import.js`, `TrainArea.asJson` of `assets/train-area.js`).
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).

use crate::js_harness::{asset, have_node, run_node_ok};

/// Run `driver_js` with the asset paths as arguments; it asserts, exits non-zero on a failure and
/// prints `ok` at the end.
fn run(driver_js: &str, assets: &[&str]) {
    let paths: Vec<_> = assets.iter().map(|a| asset(a)).collect();
    let args: Vec<_> = paths.iter().map(|p| p.as_os_str()).collect();
    run_node_ok(driver_js, &args);
}

/// A fake clock and `fetch` for `live.js` with a stub DOM.
const LIVE_KIT: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const tick = () => new Promise(r => setImmediate(r));
const flush = async () => { for (let i = 0; i < 8; i++) await tick(); };
function world(statusDataset) {
  const K = {now: 1000000, timers: [], seq: 0, sent: [], defaults: {}, first: {}, assigned: [], reloads: 0, listeners: {}};
  const reply = (status, body) => ({status, ok: status < 300, headers: {get: () => null},
    json: () => body === undefined ? Promise.reject(new Error('no JSON')) : Promise.resolve(body)});
  K.reply = reply;
  function el(tag) {
    return {tagName: tag, children: [], attrs: {}, className: '', textContent: '', hidden: false, disabled: false, dataset: {},
      style: {setProperty() {}}, classList: {add() {}, remove() {}, toggle() {}},
      setAttribute(k, v) { this.attrs[k] = v; }, appendChild(c) { return c; }, addEventListener() {}};
  }
  const status = statusDataset ? Object.assign(el('p'), {dataset: statusDataset}) : null;
  const document = {hidden: false, readyState: 'complete', cookie: '', body: el('body'), createElement: el,
    querySelector: () => null, querySelectorAll: () => [],
    getElementById: id => id === 'restart-status' ? status : null,
    addEventListener(t, f) { (K.listeners[t] = K.listeners[t] || []).push(f); }};
  const sandbox = {console, Promise, Headers, URLSearchParams, FormData, Blob, AbortController,
    location: {pathname: '/config', search: '', hash: '', origin: 'https://h', assign(u) { K.assigned.push(u); },
      reload() { K.reloads++; }, replace(u) { K.assigned.push(u); }},
    Date: {now: () => K.now}, document, history: {}, scrollX: 0, scrollY: 0, scrollTo() {}, confirm: () => true,
    innerHeight: 800, requestAnimationFrame: f => f(), getComputedStyle: () => ({position: 'static'}),
    PrismViewport: {bottomLift: () => 0, keyboardOpen: () => false},
    addEventListener() {},
    setTimeout(f, ms) { const t = {id: ++K.seq, at: K.now + ms, f}; K.timers.push(t); return t.id; },
    clearTimeout(id) { K.timers = K.timers.filter(t => t.id !== id); },
    fetch(url, init) {
      return new Promise((resolve, reject) => {
        const s = {url, init, resolve, reject};
        K.sent.push(s);
        const q = K.first[url];
        const spec = q && q.length ? q.shift() : K.defaults[url];
        if (!spec || spec[0] === 'hang') return;
        resolve(reply(spec[0], spec[1]));
      });
    }};
  sandbox.window = sandbox;
  K.advance = async (ms) => {
    const end = K.now + ms;
    for (;;) {
      const next = K.timers.filter(t => t.at <= end).sort((a, b) => a.at - b.at)[0];
      if (!next) break;
      K.timers = K.timers.filter(t => t !== next); K.now = next.at; next.f(); await flush();
    }
    K.now = end;
  };
  K.status = status;
  K.ctx = vm.createContext(sandbox);
  K.document = document;
  K.sentTo = (u) => K.sent.filter(s => s.url === u);
  return K;
}
const cfg = (generation, epoch) => [200, {config: {}, revision: 1, generation, epoch, pendingCount: 0}];
"#;

/// QA-113: the restart page leaves when the generation changes or the process does.
const RESTART_PAGE: &str = r#"
(async () => {
  const scenarios = [
    // [what the server answers after the first answers, expected to go back]
    ['same generation, same process', cfg(1, 111), false],
    ['same generation, another process (a crash restart)', cfg(1, 222), true],
    ['next generation, same process', cfg(2, 111), true],
  ];
  for (const [what, later, back] of scenarios) {
    const K = world({generation: '1', epoch: '111', target: '/config'});
    K.defaults['/v1/config'] = cfg(1, 111);
    vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), K.ctx);
    await K.advance(1600);
    assert.ok(/Waiting for the server/.test(K.status.textContent), K.status.textContent);
    assert.deepStrictEqual(K.assigned, [], 'still the old server');
    K.defaults['/v1/config'] = later;
    await K.advance(2000);
    assert.deepStrictEqual(K.assigned, back ? ['/config'] : [], what);
    if (back) assert.strictEqual(K.status.textContent, 'Server is back; reloading…', what);
  }
  // A page without data-epoch (an older server's) compares the generation only.
  const K = world({generation: '1', target: '/config'});
  K.defaults['/v1/config'] = cfg(1, 5);
  vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), K.ctx);
  await K.advance(5000);
  assert.deepStrictEqual(K.assigned, []);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

/// QA-113: Restart pressed before the first /v1/config answer reads the generation first, then
/// reloads only when the server is another generation or process.
const RESTART_BUTTON: &str = r#"
(async () => {
  const K = world(null);
  K.first['/v1/config'] = [['hang']];                 // the page's first poll is still out
  K.defaults['/config/restart'] = [200, {}];
  vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), K.ctx);
  await flush();
  const btn = {type: 'button', dataset: {}, textContent: 'Restart', disabled: false};
  (K.listeners.click || []).forEach(f => f({target: {closest: (sel) => /data-restart/.test(sel) ? btn : null}, preventDefault() {}, stopPropagation() {}}));
  await flush();
  assert.strictEqual(K.sentTo('/config/restart').length, 0, 'not posted before the generation is known');
  K.sentTo('/v1/config')[0].resolve(K.reply(...cfg(1, 7)));
  await flush();
  assert.strictEqual(K.sentTo('/config/restart').length, 1, 'posted once it is known');
  K.defaults['/v1/config'] = cfg(1, 7);
  await K.advance(3000);
  assert.strictEqual(K.reloads, 0, 'the old server still answers');
  K.defaults['/v1/config'] = cfg(1, 8);               // the process restarted: generation 1 again
  await K.advance(3000);
  assert.strictEqual(K.reloads, 1, 'another process: the page reloads');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

#[test]
fn the_restart_page_waits_for_another_generation_or_process() {
    if have_node() {
        // QA-113
        run(&format!("{LIVE_KIT}{RESTART_PAGE}"), &["live.js"]);
    }
}

#[test]
fn restart_pressed_before_the_first_config_answer_reads_the_generation_first() {
    if have_node() {
        // QA-113
        run(&format!("{LIVE_KIT}{RESTART_BUTTON}"), &["live.js"]);
    }
}

/// QA-115: the Jobs list's poll backs off through an outage and says so.
const JOBS_OUTAGE: &str = r#"
const tick = () => new Promise(r => setImmediate(r));
const W = {now: 0, seq: 0, timers: [], reqs: [], listeners: {}};
function el(extra) {
  return Object.assign({dataset: {}, style: {}, children: [], hidden: false, className: '', textContent: '', href: '',
    classList: {add() {}, remove() {}}, appendChild(c) { return c; }, append() {}, replaceChildren() {}, remove() {},
    setAttribute() {}, addEventListener() {}, querySelector: () => null, querySelectorAll: () => [],
    closest() { return el(); }}, extra);
}
const inserted = [];
const box = el({parentNode: {insertBefore(n) { inserted.push(n); }}});
const els = {'train-strip': el({dataset: {tab: 'jobs'}}), 'train-jobs': box, 'train-jobs-empty': el(),
  'train-jobs-count': el(), 'train-counts': el(), 'train-next-text': el(), 'train-next-link': el()};
const document = {hidden: false, getElementById: id => els[id] || null, querySelector: () => null, querySelectorAll: () => [],
  createElement: () => el(), createTextNode: () => el(), createDocumentFragment: () => el(),
  addEventListener(t, f) { (W.listeners['d:' + t] = W.listeners['d:' + t] || []).push(f); }};
const win = {addEventListener(t, f) { (W.listeners['w:' + t] = W.listeners['w:' + t] || []).push(f); }};
const sandbox = {console, Promise, window: win, history: {replaceState() {}}, location: {pathname: '/train/jobs', hash: ''},
  fetch(url) { return new Promise(resolve => { W.reqs.push({url, resolve}); }); },
  setTimeout(f, ms) { const t = {id: ++W.seq, at: W.now + ms, ms, f}; W.timers.push(t); return t.id; },
  clearTimeout(id) { W.timers = W.timers.filter(t => t.id !== id); },
  setInterval() { return 0; }, clearInterval() {}, document};
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);    // live.js: PrismPoll only
sandbox.document = document;
const fire = async (k) => { (W.listeners[k] || []).forEach(f => f({})); await tick(); };
const answer = async (url, status, body) => {
  const r = W.reqs.filter(x => x.url === url).pop();
  W.reqs = W.reqs.filter(x => x !== r);
  r.resolve({ok: status < 300, status, json: () => body === undefined ? Promise.reject(new Error('no JSON')) : Promise.resolve(body)});
  for (let i = 0; i < 6; i++) await tick();
};
const jobsTimer = () => W.timers.filter(t => t.ms > 0 && t.ms !== 30000 && t.ms !== 2000)[0];
const advance = async (ms) => {
  const end = W.now + ms;
  for (;;) {
    const next = W.timers.filter(t => t.at <= end).sort((a, b) => a.at - b.at)[0];
    if (!next) break;
    W.now = next.at; W.timers = W.timers.filter(t => t !== next); next.f(); await tick();
  }
  W.now = end;
};
(async () => {
  vm.runInContext(fs.readFileSync(process.argv[3], 'utf8'), ctx);  // train-area.js
  await fire('d:DOMContentLoaded');
  // The summary chain is out too; only the Jobs list is looked at here.
  await answer('/v1/train/jobs', 200, {jobs: {active: [], queued: []}});
  await answer('/v1/train/jobs', 200, {jobs: {active: [], queued: []}});   // the refresh that came meanwhile
  assert.strictEqual(jobsTimer().ms, 3000, 'a good answer: every 3 s');
  // A proxy's HTML 502, and a 500 with JSON that is not the list: the loop backs off and says so.
  await advance(3000);
  await answer('/v1/train/jobs', 502, undefined);
  assert.strictEqual(jobsTimer().ms, 6000, 'one failure: 6 s');
  assert.ok(inserted.length === 1 && /not answering/.test(inserted[0].textContent), 'the outage line');
  assert.strictEqual(inserted[0].hidden, false);
  await advance(6000);
  await answer('/v1/train/jobs', 500, {success: false});
  assert.strictEqual(jobsTimer().ms, 12000, 'two failures: 12 s');
  // The server is back: the line goes and the pace returns.
  await advance(12000);
  await answer('/v1/train/jobs', 200, {jobs: {active: [], queued: []}});
  assert.strictEqual(jobsTimer().ms, 3000);
  assert.strictEqual(inserted[0].hidden, true, 'the outage line is gone');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

#[test]
fn the_jobs_list_backs_off_and_says_so_when_the_server_does_not_answer() {
    if have_node() {
        // QA-115
        let head =
            "const assert = require('assert'); const fs = require('fs'), vm = require('vm');";
        run(
            &format!("{head}{JOBS_OUTAGE}"),
            &["live.js", "train-area.js"],
        );
    }
}

/// QA-117: 64 emoji (128 UTF-16 units) are a valid class name; 65 are refused.
const CLASS_NAMES: &str = r#"
const assert = require('assert');
require(require('path').join(require('path').dirname(process.argv[2]), 'dom.js'));   // sets PrismFetch / PrismDom
const CP = require(process.argv[2]).util;
const emoji = (n) => '\u{1F600}'.repeat(n);
assert.strictEqual(CP.validName(emoji(64)), emoji(64));
assert.strictEqual(CP.validName(emoji(65)), null);
assert.strictEqual(CP.nameProblem(emoji(64)), '');
assert.ok(/at most 64 characters/.test(CP.nameProblem(emoji(65))));
assert.strictEqual(CP.validName('a'.repeat(64)), 'a'.repeat(64));
assert.strictEqual(CP.validName('a'.repeat(65)), null);
console.log('ok');
"#;

#[test]
fn class_names_count_code_points_like_the_server() {
    if have_node() {
        // QA-117
        run(CLASS_NAMES, &["class-picker.js"]);
    }
}

/// Audit extra 37: the Import preview names the HTTP status when the answer is not JSON.
const IMPORT_PREVIEW: &str = r#"
const assert = require('assert'), fs = require('fs'), vm = require('vm');
const tick = () => new Promise(r => setImmediate(r));
function el(extra) {
  return Object.assign({dataset: {}, style: {}, children: [], hidden: false, className: '', textContent: '', value: '',
    classList: {add() {}, remove() {}, toggle() {}}, elements: {}, appendChild(c) { return c; }, setAttribute() {},
    getAttribute: () => null, addEventListener() {}, querySelector: () => null, querySelectorAll: () => []}, extra);
}
const preview = el();
const importForm = el({elements: {path: el({value: '/videos'})}});
const els = {'import-form': importForm, 'import-preview': preview};
const sent = [];
const document = {getElementById: id => els[id] || null, querySelector: () => null, querySelectorAll: () => [],
  addEventListener() {}, createElement: () => el(), readyState: 'complete', hidden: false};
// live.js's pure part (no document) for the real PrismPoll.latest.
const live = vm.createContext({});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), live);
const sandbox = {console, Promise, document, URLSearchParams, encodeURIComponent,
  window: {PrismPoll: {create: () => ({kick() {}, start() {}}), latest: live.PrismPoll.latest}, addEventListener() {}},
  location: {search: '', hash: '', pathname: '/train/import'},
  sessionStorage: {getItem: () => null, setItem() {}, removeItem() {}},
  localStorage: {getItem: () => null, setItem() {}, removeItem() {}},
  setTimeout(f) { f(); return 1; }, clearTimeout() {},
  fetch(url) {
    sent.push(url);
    return Promise.resolve({ok: false, status: 502, json: () => Promise.reject(new SyntaxError("Unexpected token '<'"))});
  }};
sandbox.window.document = document; sandbox.window.location = sandbox.location;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[3], 'utf8'), sandbox);   // train-area.js: TrainArea
vm.runInContext(fs.readFileSync(process.argv[4], 'utf8'), sandbox);   // train-import.js
(async () => {
  for (let i = 0; i < 6; i++) await tick();
  assert.strictEqual(sent.length, 1);
  assert.ok(/HTTP 502/.test(preview.textContent), preview.textContent);
  assert.ok(!/Unexpected token/.test(preview.textContent), preview.textContent);
  assert.strictEqual(preview.hidden, false, 'the user sees that the check failed');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

#[test]
fn the_import_preview_names_the_status_of_an_answer_that_is_not_json() {
    if have_node() {
        // Audit extras 37 / 38
        run(
            IMPORT_PREVIEW,
            &["live.js", "train-area.js", "train-import.js"],
        );
    }
}
