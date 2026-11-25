//! The Import, Auto-label, Train and Models tabs' scripts on a stub DOM, run in `node` with a fake
//! clock and a fake `fetch`: the Jobs card every one of them shows (`TrainArea.follow` in
//! `assets/train-area.js`: a busy run's state, step text and settings line kept current from
//! `GET /v1/training?poll=1`, Finish now and the early-stop notice, and the page moving on once
//! the run ends), the Train tab's runs line (`assets/train-train.js`) and the shared `TrainArea`
//! helpers (`jobs(query)`, the auto-label options kept for an import, the form toggles).
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).

use crate::js_harness::{have_node, run_node_ok};

/// A stub DOM with the selectors the tab scripts use (`tag`, `#id`, `.cls`, `[attr]`,
/// `[attr="v"]`, `:checked`, the descendant combinator and `,`), a fake clock behind `setTimeout`
/// and a fake `fetch` the driver answers. `process.argv[2]` is the assets folder.
const KIT: &str = r##"
const assert = require('assert'), fs = require('fs'), path = require('path'), vm = require('vm');
const ASSETS = process.argv[2];
const tick = () => new Promise(r => setImmediate(r));
const flush = async () => { for (let i = 0; i < 8; i++) await tick(); };

// One compound selector ("tr[data-run=\"a\"]", ".tr-settings", "input:checked") as a test.
function compound(s) {
  const tests = [];
  const re = /^([a-z][a-z0-9]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]|:(checked)/gi;
  let m, at = 0;
  while ((m = re.exec(s)) && m.index === at) {
    at = re.lastIndex;
    if (m[1]) { const t = m[1].toUpperCase(); tests.push(e => e.tagName === t); }
    else if (m[2]) { const id = m[2]; tests.push(e => e.id === id); }
    else if (m[3]) { const c = m[3]; tests.push(e => e.classList.contains(c)); }
    else if (m[4]) { const k = m[4], v = m[5]; tests.push(e => e.hasAttribute(k) && (v === undefined || e.getAttribute(k) === v)); }
    else if (m[6]) tests.push(e => !!e.checked);
  }
  if (at !== s.length) throw new Error('the stub DOM cannot match ' + JSON.stringify(s));
  return e => tests.every(t => t(e));
}
function matcher(sel) {
  if (sel.indexOf(',') >= 0) { const any = sel.split(',').map(matcher); return e => any.some(m => m(e)); }
  const parts = sel.trim().split(/\s+(?=(?:[^"]*"[^"]*")*[^"]*$)/).map(compound);
  return e => {
    if (!parts[parts.length - 1](e)) return false;
    let i = parts.length - 2, p = e.parentNode;
    for (; i >= 0 && p; p = p.parentNode) if (parts[i](p)) i--;
    return i < 0;
  };
}
const camel = k => k.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
const kebab = k => k.replace(/[A-Z]/g, c => '-' + c.toLowerCase());
class El {
  constructor(tag, attrs, kids) {
    this.tagName = String(tag).toUpperCase(); this.attrs = {}; this.children = []; this.parentNode = null;
    this.listeners = {}; this._text = ''; this.hidden = false; this.value = ''; this.checked = false; this.disabled = false;
    this.open = false; this.style = {};
    const cls = () => (this.attrs['class'] || '').split(/\s+/).filter(Boolean);
    this.classList = {contains: c => cls().indexOf(c) >= 0,
      add: (...c) => { this.className = cls().concat(c.filter(x => cls().indexOf(x) < 0)).join(' '); },
      remove: (...c) => { this.className = cls().filter(x => c.indexOf(x) < 0).join(' '); },
      toggle: (c, on) => { if (on === undefined ? !cls().includes(c) : on) this.classList.add(c); else this.classList.remove(c); }};
    const self = this;
    this.dataset = new Proxy({}, {
      get: (_, k) => self.attrs['data-' + kebab(String(k))],
      set: (_, k, v) => { self.attrs['data-' + kebab(String(k))] = String(v); return true; },
      has: (_, k) => ('data-' + kebab(String(k))) in self.attrs,
    });
    Object.keys(attrs || {}).forEach(k => {
      const v = attrs[k];
      if (k === 'hidden') this.hidden = !!v; else if (k === 'checked') this.checked = !!v;
      else if (k === 'value') this.value = String(v); else if (k === 'disabled') this.disabled = !!v;
      else this.attrs[k] = String(v);
    });
    (kids || []).forEach(c => this.appendChild(typeof c === 'string' ? new El('#text', {}).setText(c) : c));
  }
  setText(t) { this._text = t; return this; }
  get id() { return this.attrs.id || ''; }
  set id(v) { this.attrs.id = String(v); }
  get className() { return this.attrs['class'] || ''; }
  set className(v) { this.attrs['class'] = String(v); }
  get name() { return this.attrs.name || ''; }
  get type() { return this.attrs.type || ''; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set textContent(v) { this._text = String(v == null ? '' : v); this.children.forEach(c => { c.parentNode = null; }); this.children = []; }
  get innerHTML() { return this.textContent; }
  set innerHTML(v) { this.textContent = v; }
  get nextSibling() { const p = this.parentNode; if (!p) return null; return p.children[p.children.indexOf(this) + 1] || null; }
  get childElementCount() { return this.children.length; }
  get firstChild() { return this.children[0] || null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }
  hasAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k); }
  removeAttribute(k) { delete this.attrs[k]; }
  appendChild(c) { if (c.parentNode) c.remove(); this.children.push(c); c.parentNode = this; return c; }
  insertBefore(c, ref) { if (c.parentNode) c.remove(); const i = ref ? this.children.indexOf(ref) : -1; this.children.splice(i < 0 ? this.children.length : i, 0, c); c.parentNode = this; return c; }
  replaceChildren(...cs) { this.textContent = ''; cs.forEach(c => this.appendChild(c)); }
  remove() { if (this.parentNode) { this.parentNode.children = this.parentNode.children.filter(x => x !== this); this.parentNode = null; } }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  dispatch(t) { const ev = {type: t, target: this, preventDefault() {}}; (this.listeners[t] || []).slice().forEach(f => f(ev)); return ev; }
  all() { const out = []; const walk = e => e.children.forEach(c => { out.push(c); walk(c); }); walk(this); return out; }
  querySelectorAll(sel) { const m = matcher(sel); return this.all().filter(m); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  closest(sel) { const m = matcher(sel); for (let e = this; e; e = e.parentNode) if (m(e)) return e; return null; }
}
const h = (tag, attrs, kids) => new El(tag, attrs, kids);

// A page: `body` (an El), the scripts (assets file names) and `route(req)` -> {json, status} or
// nothing (held; req.respond(json, status) answers it later).
function page(o) {
  const W = {now: 0, seq: 0, timers: [], reqs: [], assigned: []};
  const body = o.body || h('body');
  const doc = {
    hidden: false, readyState: 'complete', cookie: '', body,
    getElementById: id => body.querySelector('#' + id) || (body.id === id ? body : null),
    querySelector: s => body.querySelector(s), querySelectorAll: s => body.querySelectorAll(s),
    createElement: t => h(t), createTextNode: t => new El('#text').setText(String(t)),
    addEventListener() {}, removeEventListener() {},
  };
  const store = () => { const m = {}; return {getItem: k => (k in m ? m[k] : null), setItem: (k, v) => { m[k] = String(v); }, removeItem: k => { delete m[k]; }}; };
  const ctx = vm.createContext({console, URL, URLSearchParams, AbortController});
  ctx.setTimeout = (f, ms) => { const t = {id: ++W.seq, at: W.now + (ms || 0), f}; W.timers.push(t); return t.id; };
  ctx.clearTimeout = id => { W.timers = W.timers.filter(t => t.id !== id); };
  ctx.setInterval = () => 0;
  W.load = name => vm.runInContext(fs.readFileSync(path.join(ASSETS, name), 'utf8'), ctx, {filename: name});
  W.load('live.js');   // before `document`: PrismPoll only
  const p = o.path || '/train/train';
  Object.assign(ctx, {
    window: ctx, document: doc, history: {replaceState() {}},
    location: {hash: '', search: o.search || '', pathname: p, origin: 'https://h',
      assign: u => W.assigned.push(u), replace: u => W.assigned.push(u), reload() {}},
    sessionStorage: store(), localStorage: store(), matchMedia: () => ({matches: false}),
    addEventListener() {}, CSS: {escape: s => String(s)}, confirm: () => true,
  });
  ctx.fetch = (url, init) => {
    init = init || {};
    const req = {url: String(url), method: init.method || 'GET', body: init.body ? String(init.body) : '', init, done: false};
    W.reqs.push(req);
    return new Promise((res, rej) => {
      req.respond = (json, status) => {
        if (req.done) return;
        req.done = true; status = status || 200;
        res({ok: status < 400, status, json: () => json === undefined ? Promise.reject(new SyntaxError('not JSON')) : Promise.resolve(JSON.parse(JSON.stringify(json)))});
      };
      if (init.signal) init.signal.addEventListener('abort', () => { if (req.done) return; req.done = true; req.aborted = true; const e = new Error('aborted'); e.name = 'AbortError'; rej(e); });
      const r = o.route ? o.route(req, W) : null;
      if (r) req.respond(r.json, r.status);
    });
  };
  W.advance = async ms => {
    const end = W.now + ms;
    for (let guard = 0; guard < 10000; guard++) {
      const next = W.timers.filter(t => t.at <= end).sort((a, b) => a.at - b.at || a.id - b.id)[0];
      if (!next) break;
      W.timers = W.timers.filter(t => t !== next); W.now = next.at; next.f(); await flush();
    }
    W.now = end; await flush();
  };
  W.ctx = ctx; W.doc = doc; W.body = body;
  (o.scripts || []).forEach(W.load);
  return W;
}
"##;

/// The Train tab's scripts, in the order `templates/train/training.html` loads them (no runs with
/// a history on these pages, so no train-metrics.js).
const TRAIN_TAB: &str = "['train-area.js', 'train-train.js']";

/// Run `KIT` + `driver` (it asserts and prints `ok` at the end).
fn run(driver: &str) {
    let assets = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    let driver = driver.replace("TRAIN_TAB", TRAIN_TAB);
    run_node_ok(&format!("{KIT}\n{driver}\n"), &[assets.as_os_str()]);
}

/// The Jobs card of a tab: a busy run's row follows the 3 s poll (state, the step text, the
/// settings line under it, Pause / Resume, Finish now, the early-stop notice), and once the run
/// has ended the page goes to its own address (not a reload, which would post the form again).
#[test]
fn a_busy_run_is_followed_until_it_ends_and_the_page_moves_on() {
    if !have_node() {
        return;
    }
    run(r##"
(async () => {
  const row = h('tr', {'data-run': 'r1'}, [
    h('td', {'data-field': 'state'}, ['running']),
    h('td', {}, [h('progress', {'data-field': 'percent', value: '5'}), h('span', {'data-field': 'describe'}, ['starting']),
      h('ul', {'data-field': 'warnings', hidden: true}),
      h('div', {'data-field': 'extend', 'data-run': 'r1', 'data-ext': '0', hidden: true}, [
        h('span', {'data-field': 'extend-text'}), h('button', {'data-field': 'extend-stop'}),
        h('button', {'data-action': 'extend-continue'})]),
      h('details', {'class': 'tr-logbox', 'data-run': 'r1', hidden: true}, [h('pre', {'data-field': 'log'})])]),
    h('td', {}, [h('form', {'data-field': 'pause'}), h('form', {'data-field': 'resume', hidden: true}),
      h('form', {'data-field': 'finish', hidden: true})])]);
  const body = h('body', {}, [h('table', {id: 'training-runs'}, [row])]);
  let answer = null;
  const run1 = (state, extra) => Object.assign({id: 'r1', state, percent: 40, stage: 'train', step: 3, steps: 6, detail: 'epoch 12/50',
    device: 'cuda:0', progress: {batch_size: 8, imgsz: 640, settings: {batch: 16, lr: 0.001, lr_schedule: 'cosine', lr_min_factor: 0.01,
      epochs: 50, patience: 10, seed: 0, scene: 'night', workers: 4},
      extension: {extension: 2, text: 'Validation still improves.', keep_epoch: 11}},
    dataset: {warnings: ['few night frames']}, log: ['a', 'b']}, extra || {});
  answer = {runs: [run1('running')], tasks: []};
  const W = page({path: '/train/train', body, scripts: TRAIN_TAB, route: req => /^\/v1\/training\?poll=1/.test(req.url) && answer ? {json: answer} : null});
  await W.advance(0);
  const polls = () => W.reqs.filter(r => /^\/v1\/training\?poll=1/.test(r.url)).length;
  assert.strictEqual(polls(), 1, 'asks at once: ' + W.reqs.map(r => r.url).join(' '));
  const f = n => row.querySelector('[data-field="' + n + '"]');
  assert.strictEqual(f('describe').textContent, 'train (3/6): epoch 12/50');
  assert.strictEqual(f('percent').value, 40);
  assert.strictEqual(row.querySelector('.tr-settings').textContent,
    'cuda:0 · batch 8 (OOM retry from 16) · 640 px · lr 1e-3 cosine → 0.01× · cap 50 ep · patience 10 · seed 0 · night · workers 4');
  assert.deepStrictEqual(f('warnings').children.map(c => c.textContent), ['few night frames']);
  assert.strictEqual(f('warnings').hidden, false);
  assert.strictEqual(f('finish').hidden, false, 'Finish now while it trains');
  assert.strictEqual(f('pause').hidden, false);
  assert.strictEqual(f('resume').hidden, true);
  assert.strictEqual(f('extend').hidden, false, 'the early-stop notice');
  assert.strictEqual(f('extend').dataset.ext, '2');
  assert.strictEqual(f('extend-stop').textContent, 'Stop now and keep epoch 11');
  assert.strictEqual(row.querySelector('details').hidden, false, 'the log box shows');
  assert.strictEqual(f('log').textContent, 'a\nb');
  // Paused: Resume, the stage named; the poll comes 3 s after the answer.
  answer = {runs: [run1('paused')], tasks: []};
  await W.advance(2999);
  assert.strictEqual(polls(), 1, 'not before 3 s');
  await W.advance(1);
  assert.strictEqual(polls(), 2);
  assert.strictEqual(f('describe').textContent, 'paused (train 3/6): epoch 12/50');
  assert.strictEqual(f('pause').hidden, true);
  assert.strictEqual(f('resume').hidden, false);
  // Continue hides that notice for this early-stop point.
  row.querySelector('[data-action="extend-continue"]').dispatch('click');
  assert.strictEqual(f('extend').hidden, true);
  await W.advance(3000);
  assert.strictEqual(f('extend').hidden, true, 'stays hidden on the next poll');
  assert.deepStrictEqual(W.assigned, [], 'nothing ended yet');
  // It ended: the page goes to its own address, and the loop stops.
  answer = {runs: [run1('done', {stage: null, progress: {}})], tasks: []};
  await W.advance(3000);
  assert.deepStrictEqual(W.assigned, ['/train/train']);
  const n = polls();
  await W.advance(30000);
  assert.strictEqual(polls(), n, 'no poll after the page moved on');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##);
}

/// Nothing busy on the page: no poll at all, and Remove with files asks first.
#[test]
fn an_idle_jobs_card_does_not_poll_and_remove_asks_first() {
    if !have_node() {
        return;
    }
    run(r##"
(async () => {
  const del = h('form', {'data-confirm': 'Delete the files of run x (1 GB) from this machine?'});
  const body = h('body', {}, [h('table', {id: 'training-runs'}, [
    h('tr', {'data-run': 'r1'}, [h('td', {'data-field': 'state'}, ['done']), h('td', {}, [del])])])]);
  const W = page({path: '/train/models', body, scripts: TRAIN_TAB});
  const asked = [];
  W.ctx.confirm = q => { asked.push(q); return false; };
  await W.advance(10000);
  assert.deepStrictEqual(W.reqs.map(r => r.url), [], 'nothing busy: nothing asked');
  let prevented = false;
  (del.listeners.submit || []).forEach(fn => fn({preventDefault() { prevented = true; }}));
  assert.deepStrictEqual(asked, ['Delete the files of run x (1 GB) from this machine?']);
  assert.ok(prevented, 'Cancel keeps the files');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##);
}

/// The Train tab's runs line: one run per ticked scene per ticked model, named before queuing;
/// several runs turn the model name field off.
#[test]
fn the_runs_line_names_each_run() {
    if !have_node() {
        return;
    }
    run(r##"
(async () => {
  const m1 = h('input', {type: 'checkbox', name: 'model', value: 'yolo11s', checked: true});
  const m2 = h('input', {type: 'checkbox', name: 'model', value: 'rfdetr-s'});
  const day = h('input', {type: 'checkbox', name: 'scene', value: 'day', 'data-prefix': 'day'});
  const night = h('input', {type: 'checkbox', name: 'scene', value: 'night', 'data-prefix': 'night'});
  const bal = h('input', {type: 'checkbox', name: 'balance', value: 'camera-day'});
  const name = h('input', {type: 'text', id: 'train-name'});
  const line = h('p', {id: 'train-runs-line', hidden: true});
  const form = h('form', {id: 'train'}, [m1, m2, day, night, bal, name, h('span', {id: 'train-name-hint'}, ['one name']), line]);
  const W = page({path: '/train/train', body: h('body', {}, [form]), scripts: TRAIN_TAB});
  await W.advance(0);
  assert.strictEqual(line.textContent, 'Queues 1 run: custom-yolo11s');
  assert.strictEqual(name.disabled, false);
  day.checked = true; night.checked = true; m2.checked = true; bal.checked = true; bal.dispatch('change');
  assert.strictEqual(line.textContent,
    'Queues 4 runs: day-yolo11s, night-yolo11s, day-rfdetr-s, night-rfdetr-s (compared on the same held-out frames); training frames balanced by camera + day');
  assert.strictEqual(name.disabled, true, 'several runs: each gets its own name');
  assert.strictEqual(form.querySelector('#train-name-hint').textContent, '4 runs: each gets its own name (<scene>-<model>-<date>).');
  m1.checked = false; m2.checked = false; m1.dispatch('change');
  assert.strictEqual(line.hidden, true, 'no model ticked: no line');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##);
}

/// `TrainArea.jobs(query)`, the one reader of `GET /v1/train/jobs` behind the Jobs tab, the Dataset
/// tab's export lines and the Refine tab's check: the list (active, then queued) and the task
/// records; an answer that is not the list (an HTTP error, a proxy's HTML, JSON without `jobs`)
/// rejects, so the poll built on it backs off.
#[test]
fn the_jobs_list_reader_resolves_the_list_and_rejects_anything_else() {
    if !have_node() {
        return;
    }
    run(r##"
(async () => {
  const answers = [
    [200, {jobs: {active: [{id: 'a'}], queued: [{id: 'q'}]}, tasks: [{id: 't'}], task: null}],
    [502, undefined],
    [200, undefined],
    [500, {success: false, message: 'x'}],
    [200, {success: true}],
    [200, {jobs: {active: []}}],
  ];
  const W = page({path: '/train/jobs', scripts: ['train-area.js'], route: () => { const a = answers.shift(); return {json: a[1], status: a[0]}; }});
  const TA = W.ctx.TrainArea;
  const ok = await TA.jobs('tasks=busy');
  assert.strictEqual(W.reqs[0].url, '/v1/train/jobs?tasks=busy');
  assert.strictEqual(W.reqs[0].init.cache, 'no-store');
  assert.deepStrictEqual(JSON.parse(JSON.stringify(ok)),
    {active: [{id: 'a'}], queued: [{id: 'q'}], all: [{id: 'a'}, {id: 'q'}], task: null, tasks: [{id: 't'}]});
  const why = [];
  for (let i = 0; i < 5; i++) why.push(await TA.jobs().then(() => 'resolved', e => e.message));
  assert.strictEqual(W.reqs[1].url, '/v1/train/jobs');
  assert.deepStrictEqual(why, ['HTTP 502', 'not JSON', 'HTTP 500', 'no job list in the answer', 'no job list in the answer']);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##);
}

/// The shared form helpers: `toggle` shows the blocks the checked radio names and disables the
/// fields of the others; the auto-label settings a run posted are kept for the run an import chains
/// (without the frames, classes and confirmations), and the defaults stand in before any run.
#[test]
fn the_form_helpers_toggle_and_keep_the_auto_label_settings() {
    if !have_node() {
        return;
    }
    run(r##"
(async () => {
  const folder = h('input', {type: 'radio', name: 'source', value: 'folder', checked: true});
  const video = h('input', {type: 'radio', name: 'source', value: 'video'});
  const a = h('div', {'data-source': 'folder'}, [h('input', {name: 'path'})]);
  const b = h('div', {'data-source': 'video both'}, [h('select', {name: 'camera'})]);
  const form = h('form', {id: 'f'}, [folder, video, a, b]);
  const W = page({path: '/train/import', body: h('body', {}, [form]), scripts: ['train-area.js']});
  const TA = W.ctx.TrainArea;
  TA.toggle(form, 'source', 'data-source');
  assert.deepStrictEqual([a.hidden, a.children[0].disabled, b.hidden, b.children[0].disabled], [false, false, true, true]);
  folder.checked = false; video.checked = true; video.dispatch('change');
  assert.deepStrictEqual([a.hidden, a.children[0].disabled, b.hidden, b.children[0].disabled], [true, true, false, false]);
  assert.strictEqual(TA.checked(form, 'source'), 'video');
  // Before any auto-label run: the form's defaults.
  assert.deepStrictEqual(JSON.parse(JSON.stringify(TA.alOptions()))[0], ['mode', 'labelers']);
  const al = h('form', {id: 'autolabel-form'});
  TA.keepAlOptions(al);
  al.listeners.formdata[0]({formData: new Map([['gdino', 'base'], ['classes', 'person'], ['limit', '50'], ['vlm', 'off'], ['confirm_large', '1']])});
  assert.deepStrictEqual(JSON.parse(JSON.stringify(TA.alOptions())), [['gdino', 'base'], ['vlm', 'off']]);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##);
}
