//! The pages' poll loops (`PrismPoll` in `assets/live.js`) and the Train area's two chains
//! (`assets/train-area.js`), run in `node` with fake timers, a fake `fetch` and a stub DOM.
//!
//! What they pin: a loop never has two requests in flight, never ticks while the tab is hidden,
//! runs at once when it shows again, backs off after errors, and a refresh of the Train area asks
//! the running chain instead of starting another (the old code started a second 3 s chain from the
//! 30 s timer and the `train-area:refresh` event, doubling the requests for good), a hung poll is
//! aborted at 15 s (an error: it backs off), and `pageshow` from the back-forward cache or `online`
//! runs a loop at once. Signed-in requests: polls carry `X-Prism-Poll: 1` (so they do not keep a
//! session alive); a 401 opens the in-page sign-in dialog (POST /login as JSON, then each refused
//! request sent once more, its caller getting that answer; polls held while it is open), and falls
//! back to the login page with the way back (`#hash` included) on Cancel, a second 401, a body that
//! cannot be sent twice, or no `<dialog>`. Native POST forms check a stale session first, and a
//! `data-confirm` on submit still asks once. `PrismDraft` keeps working without storage. The
//! config poll of every page (a `PrismPoll` loop: 2 s, backing off to 30 s with the header line,
//! hidden / focus / pageshow / online, `PrismLive.poll(true)`, a signed-out answer opening the
//! dialog); `PrismLive.ui` (a polled value refreshes an unedited field, an edited one keeps the
//! user's value with a "Changed elsewhere" note) and `PrismLive.keep` / `patch` (skip, defer while
//! the user is in the section, rebuild only changed children).
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).

use crate::js_harness::{asset, have_node, run_node_ok};

/// Run `driver_js` with the paths of `live.js` and `train-area.js` as its arguments; it asserts,
/// exits non-zero on a failure and prints `ok` at the end.
fn run(driver_js: &str) {
    run_node_ok(
        driver_js,
        &[
            asset("live.js").as_os_str(),
            asset("train-area.js").as_os_str(),
        ],
    );
}

/// Fake clock and tab for the scheduler.
const SCHEDULER: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({console, Promise});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);   // no `document`: only PrismPoll
const Poll = vm.runInContext('PrismPoll', ctx);
const tick = () => new Promise(r => setImmediate(r));

function world() {
  const w = {now: 0, hidden: false, timers: [], seq: 0, visible: [], calls: [], pending: []};
  w.env = {
    hidden: () => w.hidden,
    set: (f, ms) => { const t = {id: ++w.seq, at: w.now + ms, ms, f}; w.timers.push(t); return t.id; },
    clear: id => { w.timers = w.timers.filter(t => t.id !== id); },
    onVisible: cb => w.visible.push(cb),
  };
  // run(): a request that stays in flight until `answer` / `fail` is called.
  w.run = () => new Promise((res, rej) => { w.calls.push(w.now); w.pending.push({res, rej}); });
  w.answer = async (v) => { w.pending.shift().res(v); await tick(); };
  w.fail = async () => { w.pending.shift().rej(new Error('x')); await tick(); };
  w.advance = async (ms) => {
    const end = w.now + ms;
    for (;;) {
      const next = w.timers.filter(t => t.at <= end).sort((a, b) => a.at - b.at)[0];
      if (!next) break;
      w.timers = w.timers.filter(t => t !== next); w.now = next.at; next.f(); await tick();
    }
    w.now = end;
  };
  w.show = async (hidden) => { w.hidden = hidden; w.visible.forEach(cb => cb()); await tick(); };
  return w;
}

(async () => {
  // Back-off table: period 2000, cap 30000.
  [[0, 2000], [1, 4000], [2, 8000], [3, 16000], [4, 30000], [5, 30000], [9, 30000]].forEach(([f, ms]) =>
    assert.strictEqual(Poll.delay(2000, f, 30000), ms, 'delay after ' + f + ' failures'));

  { // A slow answer never overlaps the next run; the next one is scheduled after it.
    const w = world(), p = Poll.create({run: w.run, period: 1000, env: w.env});
    p.start(); await tick();
    await w.advance(10000);
    assert.strictEqual(w.calls.length, 1, 'one request in flight however long it takes');
    await w.answer(1);
    assert.strictEqual(w.timers.length, 1);
    assert.strictEqual(w.timers[0].ms, 1000);
    await w.advance(1000);
    assert.strictEqual(w.calls.length, 2);
  }
  { // Hidden: no timer, no request; visible: a request at once.
    const w = world(), p = Poll.create({run: w.run, period: 1000, env: w.env});
    p.start(); await tick();
    await w.answer(1);
    await w.show(true);
    assert.strictEqual(w.timers.length, 0, 'the timer is gone while hidden');
    await w.advance(60000);
    assert.strictEqual(w.calls.length, 1, 'no request while hidden');
    await w.show(false);
    assert.strictEqual(w.calls.length, 2, 'a request at once when visible');
    // hidden while a request is in flight: it finishes, nothing is scheduled
    await w.show(true);
    await w.answer(1);
    assert.strictEqual(w.timers.length, 0);
    await w.show(false);
    assert.strictEqual(w.calls.length, 3);
    // a second visibility event while one is in flight does not start a second request
    await w.show(false);
    assert.strictEqual(w.pending.length <= 2, true);
  }
  { // Errors back off, a success resets.
    const w = world(), p = Poll.create({run: w.run, period: 1000, env: w.env});
    p.start(); await tick();
    const seen = [];
    for (let i = 0; i < 6; i++) { await w.fail(); seen.push(w.timers[0].ms); await w.advance(w.timers[0].ms); }
    assert.deepStrictEqual(seen, [2000, 4000, 8000, 16000, 30000, 30000]);
    await w.answer(1);
    assert.strictEqual(w.timers[0].ms, 1000);
  }
  { // kick() while in flight: exactly one more run right after, never two at once.
    const w = world(), p = Poll.create({run: w.run, period: 1000, env: w.env});
    p.start(); await tick();
    p.kick(); p.kick(); p.kick(); await tick();
    assert.strictEqual(w.pending.length, 1);
    await w.answer(1);
    assert.strictEqual(w.pending.length, 1, 'one follow-up run');
    assert.strictEqual(w.calls.length, 2);
    await w.answer(1);
    assert.strictEqual(w.timers.length, 1, 'then one timer');
  }
  { // more() false parks the loop until kick(); a visibility change does not wake a parked loop.
    const w = world(), p = Poll.create({run: w.run, period: 1000, more: v => v === 'busy', env: w.env});
    p.start(); await tick();
    await w.answer('busy'); await w.advance(1000);
    await w.answer('idle');
    assert.strictEqual(w.timers.length, 0, 'parked: no timer');
    await w.advance(60000);
    await w.show(true); await w.show(false);
    assert.strictEqual(w.calls.length, 2);
    p.kick(); await tick();
    assert.strictEqual(w.calls.length, 3);
    p.stop(); await w.answer('busy');
    assert.strictEqual(w.timers.length, 0, 'stop() wins over a late answer');
  }
  { // later() unparks without a run now: the next one a period away (after the one in flight).
    const w = world(), p = Poll.create({run: w.run, period: 1000, more: v => v === 'busy', env: w.env});
    p.start(); await tick(); await w.answer('idle');
    p.later(); await tick();
    assert.strictEqual(w.calls.length, 1, 'no run now');
    assert.deepStrictEqual(w.timers.map(t => t.ms), [1000]);
    await w.advance(1000);
    assert.strictEqual(w.calls.length, 2);
    p.later(); p.later(); await tick();
    assert.strictEqual(w.timers.length, 0, 'in flight: nothing scheduled beside it');
    await w.answer('busy');
    assert.deepStrictEqual(w.timers.map(t => t.ms), [1000], 'then one timer');
  }
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

/// The Train area script against a stub DOM: one chain per list, whatever asks for a refresh.
const TRAIN_AREA: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const tick = () => new Promise(r => setImmediate(r));

const W = {now: 0, seq: 0, timers: [], reqs: [], log: [], listeners: {}};
function el(extra) {
  const e = {dataset: {}, style: {}, children: [], hidden: false, className: '', textContent: '', href: '',
    scrollLeft: 0, offsetLeft: 0, clientWidth: 0, offsetWidth: 0, classList: {add() {}, remove() {}},
    appendChild(c) { return c; }, append() {}, replaceChildren() {}, remove() {}, setAttribute() {},
    addEventListener() {}, querySelector: () => null, querySelectorAll: () => [],
    closest() { return el(); }, ...extra};
  return e;
}
const strip = el({dataset: {tab: 'jobs'}});
const box = el();
const els = {'train-strip': strip, 'train-jobs': box, 'train-jobs-empty': el(), 'train-jobs-count': el(),
  'train-counts': el(), 'train-next-text': el({closest: () => el()}), 'train-next-link': el()};
const document = {
  hidden: false,
  getElementById: id => els[id] || null,
  querySelector: () => null,
  createElement: () => el(), createTextNode: () => el(), createDocumentFragment: () => el(),
  addEventListener(t, f) { (W.listeners['d:' + t] = W.listeners['d:' + t] || []).push(f); },
};
const win = {addEventListener(t, f) { (W.listeners['w:' + t] = W.listeners['w:' + t] || []).push(f); }};
const fire = async (k) => { (W.listeners[k] || []).forEach(f => f({})); await tick(); };
const body = {
  '/v1/train/summary': {summary: {counted: true, frames_collected: 1, runs: 0, models: 0, exports: 0},
    next: {text: 't', label: 'l', link: '/x', tab: 'jobs', key: 'k'}},
  '/v1/train/jobs': {jobs: {active: [], queued: []}},
};
const sandbox = {
  console, Promise, window: win, history: {replaceState() {}}, location: {pathname: '/train/jobs', hash: ''},
  fetch(url) {
    return new Promise(resolve => { W.reqs.push({url, resolve}); W.log.push(url); });
  },
  setTimeout(f, ms) { const t = {id: ++W.seq, at: W.now + ms, ms, f}; W.timers.push(t); return t.id; },
  clearTimeout(id) { W.timers = W.timers.filter(t => t.id !== id); },
  setInterval(f, ms) { const t = {id: ++W.seq, at: W.now + ms, ms, f, every: ms}; W.timers.push(t); return t.id; },
  clearInterval(id) { W.timers = W.timers.filter(t => t.id !== id); },
};
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);    // PrismPoll (no document yet)
sandbox.document = document;
const inflight = (u) => W.reqs.filter(r => r.url === u).length;
const answerAll = async () => {
  const rs = W.reqs.splice(0);
  rs.forEach(r => r.resolve({ok: true, json: () => Promise.resolve(body[r.url])}));
  await tick(); await tick();
};
const advance = async (ms) => {
  const end = W.now + ms;
  for (;;) {
    const next = W.timers.filter(t => t.at <= end).sort((a, b) => a.at - b.at)[0];
    if (!next) break;
    W.now = next.at;
    if (next.every) next.at += next.every; else W.timers = W.timers.filter(t => t !== next);
    next.f(); await tick();
  }
  W.now = end;
};
const jobsTimers = () => W.timers.filter(t => t.ms === 3000).length;
const count = (u) => W.log.filter(x => x === u).length;

(async () => {
  vm.runInContext(fs.readFileSync(process.argv[3], 'utf8'), ctx);  // train-area.js
  await fire('d:DOMContentLoaded');
  // A slow server: refreshes arrive while the jobs answer is still out.
  await fire('w:train-area:refresh'); await fire('w:train-area:refresh');
  assert.ok(inflight('/v1/train/jobs') <= 1, 'one jobs request in flight, got ' + inflight('/v1/train/jobs'));
  assert.ok(inflight('/v1/train/summary') <= 1, 'one summary request in flight');
  for (let i = 0; i < 4; i++) await answerAll();
  assert.strictEqual(W.reqs.length, 0);
  assert.strictEqual(jobsTimers(), 1, 'exactly one 3 s chain, got ' + jobsTimers());

  // The 30 s timer lands while a jobs answer is out.
  const before = count('/v1/train/jobs');
  await advance(3000);                        // the chain's request goes out, unanswered
  assert.strictEqual(inflight('/v1/train/jobs'), 1);
  await advance(27000);                       // ... and the 30 s timer fires meanwhile
  assert.ok(inflight('/v1/train/jobs') <= 1, 'still one in flight after the 30 s timer');
  for (let i = 0; i < 4; i++) await answerAll();
  assert.strictEqual(jobsTimers(), 1, 'still one chain after the 30 s timer, got ' + jobsTimers());

  // Steady state: one request per 3 s, however long it runs.
  const t0 = count('/v1/train/jobs');
  for (let i = 0; i < 20; i++) { await advance(3000); await answerAll(); }
  const n = count('/v1/train/jobs') - t0;
  assert.ok(n >= 19 && n <= 22, 'about 20 jobs requests in 60 s, got ' + n);

  // Hidden: nothing goes out, however long.
  document.hidden = true; await fire('d:visibilitychange');
  await answerAll();
  const h0 = W.log.length;
  await advance(120000); await answerAll();
  assert.strictEqual(W.log.length, h0, 'no request while hidden');
  // Visible again: the lists reload at once.
  document.hidden = false; await fire('d:visibilitychange');
  assert.ok(W.log.length > h0, 'a request at once when visible');
  await answerAll();
  assert.strictEqual(jobsTimers(), 1);

  // The count is still being made: the summary asks again every 2 s, one at a time.
  body['/v1/train/summary'].summary.counted = false;
  await fire('w:train-area:refresh'); await answerAll();
  const s0 = count('/v1/train/summary');
  await advance(2000); assert.strictEqual(inflight('/v1/train/summary'), 1);
  await advance(10000); assert.strictEqual(inflight('/v1/train/summary'), 1, 'no overlap');
  await answerAll();
  assert.strictEqual(count('/v1/train/summary') - s0, 1);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

/// The fetch wrapper of `live.js` with a fake `fetch` and `location`.
const AUTH: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const tick = () => new Promise(r => setImmediate(r));
const sent = [], assigned = [], replies = [];
const location = {pathname: '/config', search: '?x=1', origin: 'https://192.168.1.5:32169',
  assign(u) { assigned.push(u); }};
// A fake clock: the redirect's "once" window and the release of waiting requests run on it.
const clock = {now: 1000000, timers: [], seq: 0};
const advance = async (ms) => {
  const end = clock.now + ms;
  for (;;) {
    const next = clock.timers.filter(t => t.at <= end).sort((a, b) => a.at - b.at)[0];
    if (!next) break;
    clock.timers = clock.timers.filter(t => t !== next); clock.now = next.at; next.f(); await tick();
  }
  clock.now = end;
};
const sandbox = {
  console, Promise, Headers, location, Date: {now: () => clock.now},
  setTimeout(f, ms) { const t = {id: ++clock.seq, at: clock.now + ms, f}; clock.timers.push(t); return t.id; },
  clearTimeout(id) { clock.timers = clock.timers.filter(t => t.id !== id); },
  fetch(url, init) {
    sent.push({url, init});
    const status = replies.length ? replies.shift() : 200;
    return Promise.resolve({status, ok: status < 400, json: () => Promise.resolve({})});
  },
};
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const fetch = (u, i) => vm.runInContext('fetch', ctx)(u, i);
const Poll = vm.runInContext('PrismPoll', ctx);
const header = (s) => {
  const h = s.init && s.init.headers;
  if (!h) return undefined;
  return typeof h.get === 'function' ? h.get('X-Prism-Poll') : h['X-Prism-Poll'];
};
const settled = async (p) => { let done = false; p.then(() => { done = true; }); await tick(); await tick(); return done; };

(async () => {
  // A request the user makes: no poll header, the caller's headers untouched.
  const mine = {headers: {Accept: 'application/json'}};
  await fetch('/v1/training', mine);
  assert.strictEqual(header(sent.pop()), undefined);
  assert.deepStrictEqual(mine.headers, {Accept: 'application/json'}, 'the caller\'s object is not changed');
  // Polls: PrismPoll.fetch, a PrismPoll run, and background().
  await Poll.fetch('/v1/config', {headers: {Accept: 'application/json'}});
  let s = sent.pop();
  assert.strictEqual(header(s), '1');
  assert.strictEqual(s.init.headers.Accept, 'application/json');
  const env = {hidden: () => false, set: () => 1, clear() {}, onVisible() {}};
  const loop = Poll.create({run: () => fetch('/v1/train/jobs'), period: 1000, env});
  loop.start(); await tick();
  assert.strictEqual(header(sent.pop()), '1', 'a PrismPoll run is a poll');
  loop.stop();
  Poll.background(() => { fetch('/v1/review/next', {headers: new Headers({Accept: 'x'})}); });
  s = sent.pop();
  assert.strictEqual(header(s), '1', 'Headers objects too');
  assert.strictEqual(s.init.headers.get('Accept'), 'x');
  await fetch('/after');
  assert.strictEqual(header(sent.pop()), undefined, 'background() ends with its function');
  // Another origin never gets the header (it would need a CORS preflight).
  Poll.background(() => { fetch('https://example.test/x'); fetch('//example.test/y'); });
  assert.strictEqual(header(sent.pop()), undefined);
  assert.strictEqual(header(sent.pop()), undefined);
  Poll.background(() => { fetch('https://192.168.1.5:32169/v1/x'); });
  assert.strictEqual(header(sent.pop()), '1', 'an absolute URL of this origin is a poll');

  // A fetch chained after a poll's answer (inside a `.then`) is not synchronous any more: through
  // PrismPoll.background (or PrismPoll.fetch) it is still a poll; a bare one is not, which is
  // why every poll path in the pages uses the helpers.
  sent.length = 0;
  const chained = Poll.create({env, period: 1000, run: () => fetch('/v1/benchmark?only=run').then(() => {
    Poll.background(() => { fetch('/v1/benchmark'); });
    fetch('/bare');
    return Poll.fetch('/v1/resources');
  })});
  chained.start(); await tick(); await tick(); await tick();
  chained.stop();
  const tag = (u) => header(sent.find(x => x.url === u));
  assert.strictEqual(tag('/v1/benchmark?only=run'), '1');
  assert.strictEqual(tag('/v1/benchmark'), '1', 'background() inside a .then');
  assert.strictEqual(tag('/v1/resources'), '1', 'PrismPoll.fetch inside a .then');
  assert.strictEqual(tag('/bare'), undefined, 'a bare chained fetch is not tagged');
  // A reload or navigation a poll makes: the poll cookie first, then the load.
  ctx.document = {cookie: ''};
  const navs = [];
  location.reload = () => navs.push('reload');
  location.replace = (u) => navs.push('replace ' + u);
  Poll.navigate(null);
  Poll.navigate('/train/jobs', true);
  Poll.navigate('/x');
  assert.ok(/^__Host-bop_poll=1; Secure; SameSite=Strict; Path=\/; Max-Age=\d+$/.test(ctx.document.cookie), ctx.document.cookie);
  assert.deepStrictEqual(navs, ['reload', 'replace /train/jobs']);
  assert.deepStrictEqual(assigned.splice(0), ['/x']);
  delete ctx.document;

  // A 503 (sign-in closed) is the page's to show.
  replies.push(503);
  const r = await fetch('/v1/training');
  assert.strictEqual(r.status, 503);
  assert.deepStrictEqual(assigned, []);
  // A 401 elsewhere: no redirect.
  replies.push(401);
  assert.strictEqual((await fetch('https://example.test/x')).status, 401);
  assert.deepStrictEqual(assigned, []);
  // A 401 of this server on a page that cannot show the sign-in dialog (no `document`, no
  // <dialog>): to the login page with the way back (path, query and #hash), once; the request
  // does not settle, so a poll loop does not back off or report an error meanwhile.
  location.hash = '#acct';
  replies.push(401, 401);
  const errors = [];
  const looped = Poll.create({run: () => fetch('/v1/train/summary'), period: 1000, env,
    onError: (n) => errors.push(n)});
  looped.start();
  const refusedOne = fetch('/v1/config');
  assert.strictEqual(await settled(refusedOne), false, 'a 401 does not settle');
  await tick();
  assert.deepStrictEqual(assigned, ['/login?next=%2Fconfig%3Fx%3D1%23acct']);
  assert.deepStrictEqual(errors, [], 'no back-off');
  assert.strictEqual(looped.state().busy, true, 'the loop waits for the page to go');
  // Still on the page 5 s later (a leave prompt kept it): the waiting requests get their 401,
  // and the next 401 goes to the login page again (no latch left set).
  await advance(4999);
  assert.strictEqual(await settled(refusedOne), false);
  await advance(1);
  assert.strictEqual((await refusedOne).status, 401, 'released with its 401');
  await tick();
  assert.strictEqual(looped.state().busy, false, 'the loop goes on');
  looped.stop();
  replies.push(401);
  fetch('/v1/training');
  await tick(); await tick();
  assert.strictEqual(assigned.length, 2, 'a later 401 redirects again');
  assigned.splice(0);
  // On the login page itself nothing loops back to it.
  const ctx2 = vm.createContext({console, Promise, Headers,
    location: {pathname: '/login', search: '', origin: 'https://h', assign(u) { assigned.push(u); }},
    fetch: () => Promise.resolve({status: 401})});
  vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx2);
  vm.runInContext('fetch', ctx2)('/v1/x');
  await tick();
  assert.strictEqual(assigned.length, 0);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

/// What every driver below shares: a fake clock with timers, a fake `fetch` answering from
/// per-URL queues (a request with a signal rejects when it aborts), and a stub DOM element with
/// a fake `<dialog>` (`showModal` / `close`).
const KIT: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const tick = () => new Promise(r => setImmediate(r));
const flush = async () => { for (let i = 0; i < 8; i++) await tick(); };
const K = {now: 1000000, timers: [], seq: 0, sent: [], queues: {}, defaults: {}, assigned: [], shown: 0, focused: null};
const advance = async (ms) => {
  const end = K.now + ms;
  for (;;) {
    const next = K.timers.filter(t => t.at <= end).sort((a, b) => a.at - b.at)[0];
    if (!next) break;
    K.timers = K.timers.filter(t => t !== next); K.now = next.at; next.f(); await flush();
  }
  K.now = end;
};
function reply(status, body, headers) {
  return {status, ok: status >= 200 && status < 300,
    headers: {get: (k) => headers && Object.prototype.hasOwnProperty.call(headers, k) ? headers[k] : null},
    json: () => body === undefined ? Promise.reject(new Error('no JSON')) : Promise.resolve(body)};
}
// queues[url]: [[status, body, headers?] | ['hang'] | ['fail']]; then defaults[url]; else in flight
// for good (`answer(s, ...)` settles one later).
function fakeFetch(url, init) {
  return new Promise((resolve, reject) => {
    const s = {url, init, resolve, reject};
    K.sent.push(s);
    if (init && init.signal) init.signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), {name: 'AbortError'})));
    const q = K.queues[url];
    const spec = q && q.length ? q.shift() : K.defaults[url];
    if (!spec || spec[0] === 'hang') return;
    if (spec[0] === 'fail') { reject(new TypeError('Failed to fetch')); return; }
    resolve(reply(spec[0], spec[1], spec[2]));
  });
}
const answer = (s, status, body, headers) => s.resolve(reply(status, body, headers));
const sentTo = (u) => K.sent.filter(s => s.url === u);
const pollMark = (s) => { const h = s.init && s.init.headers; return h && (typeof h.get === 'function' ? h.get('X-Prism-Poll') : h['X-Prism-Poll']); };
function el(tag) {
  const e = {tagName: tag.toUpperCase(), children: [], attrs: {}, listeners: {}, className: '', textContent: '',
    hidden: false, value: '', disabled: false, parentNode: null,
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    hasAttribute(k) { return k in this.attrs; },
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter(g => g !== f); },
    focus() { K.focused = this; }};
  if (tag === 'dialog') {
    e.open = false;
    e.showModal = function () { if (!this.parentNode) throw new Error('not in the document'); if (this.open) throw new Error('open'); this.open = true; K.shown++; };
    e.close = function () { this.open = false; };
  }
  return e;
}
function find(root, pred) {
  for (const c of root.children || []) { if (pred(c)) return c; const f = find(c, pred); if (f) return f; }
  return null;
}
function makeDocument() {
  const signed = el('span'); signed.textContent = ' tester ';
  const d = {hidden: false, readyState: 'complete', cookie: '', body: el('body'), listeners: {},
    createElement: el, querySelector: s => s === '.signed-in' ? signed : null, querySelectorAll: () => [],
    getElementById: () => null,
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter(g => g !== f); },
    fire(t) { (this.listeners[t] || []).slice().forEach(f => f({})); }};
  return d;
}
function sandboxOf(location) {
  return {console, Promise, Headers, URLSearchParams, FormData, Blob, AbortController, location,
    Date: {now: () => K.now},
    setTimeout(f, ms) { const t = {id: ++K.seq, at: K.now + ms, f}; K.timers.push(t); return t.id; },
    clearTimeout(id) { K.timers = K.timers.filter(t => t.id !== id); },
    fetch: fakeFetch};
}
// The sign-in dialog's parts.
function dialogOf(doc) {
  const dlg = doc.body.children.find(c => c.tagName === 'DIALOG');
  if (!dlg) return null;
  return {dlg, form: find(dlg, e => e.tagName === 'FORM'),
    user: find(dlg, e => e.attrs.name === 'username'), pass: find(dlg, e => e.attrs.name === 'password'),
    msg: find(dlg, e => /^signin-msg/.test(e.className)),
    cancel: find(dlg, e => e.tagName === 'BUTTON' && e.type === 'button')};
}
// Submit the dialog's form; true when it kept the browser from navigating.
function submitDialog(ui) {
  let prevented = false;
  ui.form.listeners.submit.forEach(h => h({preventDefault() { prevented = true; }}));
  return prevented;
}
"#;

/// The sign-in dialog of `live.js` (the 401 path of the fetch wrapper) against a stub DOM.
const DIALOG: &str = r#"
const location = {pathname: '/train/review', search: '?auto=overlaps', hash: '#f1',
  origin: 'https://192.168.1.5:32169', assign(u) { K.assigned.push(u); }};
const sandbox = sandboxOf(location);
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);   // no document yet: PrismPoll only
const document = makeDocument();
sandbox.document = document;
sandbox.HTMLDialogElement = function () {};
const fetch = (u, i) => vm.runInContext('fetch', ctx)(u, i);
const Poll = vm.runInContext('PrismPoll', ctx);
const env = {hidden: () => false, set: () => 1, clear() {}, onVisible() {}};
const BACK = '/login?next=' + encodeURIComponent('/train/review?auto=overlaps#f1');

(async () => {
  // A refused save: the dialog opens over the page and the request waits behind it.
  K.queues['/v1/review/labels?frame=a'] = [[401, {success: false}], [200, {success: true, saved: 1}]];
  const body1 = JSON.stringify({boxes: [1]});
  let r1 = null;
  fetch('/v1/review/labels?frame=a', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: body1}).then(r => { r1 = r; });
  await flush();
  const ui = dialogOf(document);
  assert.ok(ui && ui.dlg.open, 'the dialog opens');
  assert.strictEqual(K.shown, 1);
  assert.strictEqual(r1, null, 'the request waits');
  assert.deepStrictEqual(K.assigned, [], 'the page stays');
  assert.strictEqual(ui.user.value, 'tester', 'the name from the header');
  assert.strictEqual(ui.user.attrs.autocomplete, 'username');
  assert.strictEqual(ui.user.attrs.autocapitalize, 'none');
  assert.strictEqual(ui.user.attrs.spellcheck, 'false');
  assert.strictEqual(ui.pass.attrs.type, 'password');
  assert.strictEqual(ui.pass.attrs.autocomplete, 'current-password');
  assert.strictEqual(K.focused, ui.pass, 'the name is known: the password field has the focus');
  assert.ok(ui.form.hasAttribute('data-no-auth-check'));
  // A second refused request (a FormData body) waits behind the same dialog.
  K.queues['/v1/review/accept-auto'] = [[401, {}], [200, {success: true, n: 2}]];
  const fd = new FormData(); fd.append('limit', '5');
  let r2 = null;
  fetch('/v1/review/accept-auto', {method: 'POST', body: fd}).then(r => { r2 = r; });
  await flush();
  assert.strictEqual(K.shown, 1, 'one dialog for both');
  // Polls wait while it is open: nothing goes out.
  const before = K.sent.length;
  let rp = null;
  Poll.fetch('/v1/config', {headers: {Accept: 'application/json'}}).then(r => { rp = r; });
  K.defaults['/v1/train/jobs'] = [200, {jobs: 1}];
  const loop = Poll.create({run: () => fetch('/v1/train/jobs'), period: 1000, env});
  loop.start(); await flush();
  assert.strictEqual(K.sent.length, before, 'no poll request while the dialog is open');
  assert.strictEqual(loop.state().busy, true, 'the loop waits');
  assert.strictEqual(rp, null);

  // A wrong password: the server's words in the dialog, which stays; the password is cleared.
  K.queues['/login'] = [[401, {success: false, error: 'Wrong username or password.'}]];
  ui.pass.value = 'nope';
  assert.ok(submitDialog(ui), 'the dialog form never navigates');
  await flush();
  assert.strictEqual(sentTo('/login').length, 1);
  const li = sentTo('/login')[0].init;
  assert.strictEqual(li.method, 'POST');
  assert.strictEqual(li.credentials, 'same-origin');
  assert.strictEqual(li.headers.Accept, 'application/json');
  assert.strictEqual(li.headers['Content-Type'], 'application/x-www-form-urlencoded');
  assert.strictEqual(pollMark(sentTo('/login')[0]), undefined, 'signing in is no poll');
  assert.strictEqual(String(li.body), 'username=tester&password=nope&next=');
  assert.strictEqual(ui.msg.textContent, 'Wrong username or password.');
  assert.strictEqual(ui.msg.hidden, false);
  assert.ok(ui.dlg.open);
  assert.strictEqual(ui.pass.value, '', 'a wrong password is cleared');
  // Too many attempts: the wait is shown, the password kept.
  K.queues['/login'] = [[429, {success: false, error: 'Too many attempts. Wait 30 s, then try again.'}]];
  ui.pass.value = 'secret'; submitDialog(ui); await flush();
  assert.strictEqual(ui.msg.textContent, 'Too many attempts. Wait 30 s, then try again.');
  assert.strictEqual(ui.pass.value, 'secret');
  // No answer at all.
  K.queues['/login'] = [['fail']];
  submitDialog(ui); await flush();
  assert.ok(/did not answer/.test(ui.msg.textContent), ui.msg.textContent);
  // Empty fields: nothing is sent.
  const n0 = sentTo('/login').length;
  ui.pass.value = ''; submitDialog(ui); await flush();
  assert.strictEqual(sentTo('/login').length, n0);
  assert.ok(ui.dlg.open && r1 === null && r2 === null);

  // Signed in: each waiting request goes out once more and its caller gets that answer; the
  // held polls go out after it.
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  K.defaults['/v1/config'] = [200, {config: {}}];
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.strictEqual(ui.dlg.open, false, 'closed');
  assert.strictEqual(ui.pass.value, '', 'the password does not stay in the page');
  assert.strictEqual(r1.status, 200);
  assert.deepStrictEqual(await r1.json(), {success: true, saved: 1});
  const again = sentTo('/v1/review/labels?frame=a');
  assert.strictEqual(again.length, 2, 'sent once more, only once');
  assert.strictEqual(again[1].init.body, body1, 'the same body');
  assert.strictEqual(again[1].init.method, 'POST');
  assert.deepStrictEqual(await r2.json(), {success: true, n: 2});
  assert.strictEqual(sentTo('/v1/review/accept-auto')[1].init.body, fd);
  assert.ok(rp && rp.status === 200, 'the held poll went out');
  assert.strictEqual(sentTo('/v1/config').length, 1);
  assert.strictEqual(pollMark(sentTo('/v1/config')[0]), '1', 'still a poll');
  assert.strictEqual(sentTo('/v1/train/jobs').length, 1);
  assert.strictEqual(loop.state().busy, false, 'the loop goes on');
  loop.stop();
  assert.deepStrictEqual(K.assigned, []);

  // A 401 again right after signing in (the cookie did not take): the login page.
  K.queues['/v1/a'] = [[401, {}], [401, {}]];
  let ra = null;
  fetch('/v1/a').then(r => { ra = r; });
  await flush();
  assert.ok(ui.dlg.open, 'the dialog again');
  assert.strictEqual(K.shown, 2);
  assert.strictEqual(K.focused, ui.pass);
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.deepStrictEqual(K.assigned, [BACK], 'the way back keeps the #hash');
  assert.strictEqual(ra, null);

  // Cancel: the dialog closes and the page goes to /login with the way back. Still here 5 s
  // later (a leave prompt kept it): the request gets its 401; the next 401 opens the dialog again.
  await advance(10000); assert.ok(ra && ra.status === 401);
  K.assigned.splice(0);
  K.queues['/v1/b'] = [[401, {}]];
  let rb = null;
  fetch('/v1/b').then(r => { rb = r; });
  await flush();
  assert.ok(ui.dlg.open);
  ui.cancel.listeners.click.forEach(h => h({}));
  assert.strictEqual(ui.dlg.open, false);
  assert.deepStrictEqual(K.assigned, [BACK]);
  await flush();
  assert.strictEqual(rb, null, 'waits for the page to go');
  await advance(5000);
  assert.strictEqual(rb.status, 401, 'released with its 401');
  // Esc is Cancel.
  K.assigned.splice(0);
  K.queues['/v1/c'] = [[401, {}]];
  fetch('/v1/c'); await flush();
  assert.ok(ui.dlg.open, 'opens again after a cancel');
  let escPrevented = false;
  ui.dlg.listeners.cancel.forEach(h => h({preventDefault() { escPrevented = true; }}));
  assert.ok(escPrevented && !ui.dlg.open);
  assert.deepStrictEqual(K.assigned, [BACK]);
  await advance(10000);

  // A body that cannot be sent twice (a stream): the login page, no dialog.
  K.assigned.splice(0);
  const shown0 = K.shown;
  K.queues['/v1/d'] = [[401, {}]];
  fetch('/v1/d', {method: 'POST', body: {getReader() {}}}); await flush();
  assert.strictEqual(K.shown, shown0);
  assert.deepStrictEqual(K.assigned, [BACK]);
  await advance(10000);

  // A hidden tab: the dialog waits until the page is visible again.
  K.assigned.splice(0);
  document.hidden = true;
  K.queues['/v1/e'] = [[401, {}], [200, {e: 1}]];
  let re = null;
  fetch('/v1/e').then(r => { re = r; });
  await flush();
  assert.strictEqual(ui.dlg.open, false, 'not while hidden');
  document.fire('visibilitychange');
  assert.strictEqual(ui.dlg.open, false);
  document.hidden = false;
  document.fire('visibilitychange');
  assert.ok(ui.dlg.open, 'opens when visible');
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.deepStrictEqual(await re.json(), {e: 1});
  assert.deepStrictEqual(K.assigned, []);

  // signIn(): a caller with nothing to repeat (a native form) waits for the dialog.
  let done = false;
  Poll.signIn().then(() => { done = true; });
  await flush();
  assert.ok(ui.dlg.open && !done);
  assert.strictEqual(Poll.signingIn(), true);
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.ok(done && !ui.dlg.open);
  assert.strictEqual(Poll.signingIn(), false);

  // A hung poll is aborted at 15 s: an error, so its loop backs off; a quick one leaves no timer.
  K.queues['/v1/slow'] = [['hang']];
  const errors = [];
  const slow = Poll.create({run: () => fetch('/v1/slow'), period: 1000, env, onError: n => errors.push(n)});
  slow.start(); await flush();
  assert.ok(sentTo('/v1/slow')[0].init.signal, 'a poll has a signal');
  await advance(14999);
  assert.deepStrictEqual(errors, []);
  await advance(1);
  assert.deepStrictEqual(errors, [1], 'aborted at 15 s and counted as an error');
  slow.stop();
  const t0 = K.timers.length;
  await Poll.fetch('/v1/config');
  assert.strictEqual(K.timers.length, t0, 'the timeout is cleared by the answer');
  // The default environment: back from the back-forward cache (`pageshow` persisted) or
  // `online`, a loop runs at once, unless a run is out (it answers or times out by itself).
  const wakeL = {};
  sandbox.addEventListener = (t, f) => { (wakeL[t] = wakeL[t] || []).push(f); };
  const fire = async (t, ev) => { (wakeL[t] || []).forEach(f => f(ev || {})); await flush(); };
  K.defaults['/v1/w'] = [200, {}];
  const woke = Poll.create({run: () => fetch('/v1/w'), period: 60000});
  woke.start(); await flush();
  assert.strictEqual(sentTo('/v1/w').length, 1);
  await fire('pageshow', {persisted: false});
  assert.strictEqual(sentTo('/v1/w').length, 1, 'a normal pageshow: nothing');
  await fire('pageshow', {persisted: true});
  assert.strictEqual(sentTo('/v1/w').length, 2, 'restored from the cache: once');
  await fire('online');
  assert.strictEqual(sentTo('/v1/w').length, 3, 'online: once');
  K.queues['/v1/w'] = [['hang']];
  await fire('online');
  await fire('online'); await fire('pageshow', {persisted: true});
  assert.strictEqual(sentTo('/v1/w').length, 4, 'never two in flight');
  woke.stop();
  await advance(15000);
  // A user request has no timeout; a caller's own signal is left alone.
  K.queues['/v1/slow'] = [['hang']];
  fetch('/v1/slow');
  assert.strictEqual(sentTo('/v1/slow')[1].init, undefined);
  const own = new AbortController();
  Poll.fetch('/v1/config', {signal: own.signal});
  assert.strictEqual(sentTo('/v1/config').slice(-1)[0].init.signal, own.signal);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

/// Without `<dialog>` the 401 still goes to the login page (with the way back).
const NO_DIALOG: &str = r#"
const location = {pathname: '/config', search: '', hash: '#models',
  origin: 'https://h', assign(u) { K.assigned.push(u); }};
const sandbox = sandboxOf(location);
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const document = makeDocument();
sandbox.document = document;          // a document, but no HTMLDialogElement
(async () => {
  K.queues['/v1/x'] = [[401, {}]];
  vm.runInContext('fetch', ctx)('/v1/x');
  await flush();
  assert.deepStrictEqual(K.assigned, ['/login?next=%2Fconfig%23models']);
  assert.strictEqual(document.body.children.length, 0, 'no dialog built');
  // A dialog element without showModal (an old browser): the same.
  sandbox.HTMLDialogElement = function () {};
  document.createElement = (t) => { const e = el(t); delete e.showModal; return e; };
  K.now += 10000; K.assigned.splice(0);
  K.queues['/v1/x'] = [[401, {}]];
  vm.runInContext('fetch', ctx)('/v1/x');
  await flush();
  assert.deepStrictEqual(K.assigned, ['/login?next=%2Fconfig%23models']);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

/// The page part of `live.js` (the config poll, native POST forms, pageshow / online) with a stub
/// DOM.
const PAGE: &str = r#"
const location = {pathname: '/config', search: '', hash: '#acct', origin: 'https://h',
  assign(u) { K.assigned.push(u); }, reload() {}, replace() {}};
const sandbox = sandboxOf(location);
const document = makeDocument();
const winL = {};
Object.assign(sandbox, {document, HTMLDialogElement: function () {}, history: {}, scrollX: 0, scrollY: 0,
  scrollTo() {}, confirm: () => true,
  addEventListener(t, f) { (winL[t] = winL[t] || []).push(f); }});
sandbox.window = sandbox;
const fireWin = async (t, ev) => { (winL[t] || []).forEach(f => f(ev || {})); await flush(); };
const FULL = [200, {config: {}, revision: 1, epoch: 'e', pendingCount: 0}];
const MINIMAL = [200, {success: true, revision: 1, epoch: 'e', pendingCount: 0}];
K.queues['/v1/config'] = [FULL];
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
// A native POST form; `listeners.submit` holds the page's own handlers (a data-confirm one), run
// after the window's capture listener unless it stopped the event; the browser posts when nobody
// prevented it. requestSubmit dispatches again, as a browser does.
let confirms = 0;
function form(method, attrs) {
  const f = el('form');
  f.setAttribute('method', method);
  Object.keys(attrs || {}).forEach(k => f.setAttribute(k, attrs[k]));
  f.requested = []; f.posted = 0;
  f.addEventListener('submit', () => { confirms++; });
  f.requestSubmit = function (by) { this.requested.push(by); return dispatch(this, by); };
  return f;
}
function dispatch(f, by) {
  let prevented = false, stopped = false;
  const ev = {target: f, submitter: by, preventDefault() { prevented = true; }, stopImmediatePropagation() { stopped = true; }};
  (winL.submit || []).forEach(h => { if (!stopped) h(ev); });
  if (!stopped) (f.listeners.submit || []).forEach(h => h(ev));
  if (!prevented) f.posted++;
  return {prevented, stopped};
}

(async () => {
  await flush();
  assert.strictEqual(sentTo('/v1/config').length, 1, 'the first poll');
  // Fresh (a signed-in answer under 60 s ago): a form posts as it always did.
  const btn = el('button');
  const f1 = form('post');
  let d = dispatch(f1, btn);
  assert.ok(!d.prevented && !d.stopped && f1.posted === 1 && confirms === 1);
  // GET forms, `data-no-auth-check` ones are never held.
  K.now += 61000;
  const n0 = K.sent.length;
  dispatch(form('get'), btn);
  dispatch(form('post', {'data-no-auth-check': ''}), btn);
  assert.strictEqual(K.sent.length, n0);
  // Stale: the submit waits for one poll-style GET /v1/config; signed in, the same button goes
  // on, and the page's own submit handler (a data-confirm) runs once, on that submit.
  confirms = 0;
  K.queues['/v1/config'].push(FULL);
  const f2 = form('POST');
  d = dispatch(f2, btn);
  assert.ok(d.prevented && d.stopped, 'held');
  assert.strictEqual(confirms, 0, 'the page handlers wait');
  assert.strictEqual(f2.posted, 0);
  await flush();
  const check = sentTo('/v1/config').slice(-1)[0];
  assert.strictEqual(pollMark(check), '1', 'a poll: it does not keep the session alive');
  assert.strictEqual(check.init.headers['If-None-Match'], undefined);
  assert.deepStrictEqual(f2.requested, [btn], 'requestSubmit with the same button');
  assert.strictEqual(f2.posted, 1);
  assert.strictEqual(confirms, 1, 'asked once');
  // Fresh again after that answer.
  dispatch(form('post'), btn);
  assert.strictEqual(sentTo('/v1/config').length, 2);

  // Stale and signed out: the dialog first, then the form goes on.
  K.now += 61000; confirms = 0;
  K.queues['/v1/config'].push(MINIMAL);
  const f3 = form('post');
  dispatch(f3, btn); await flush();
  const ui = dialogOf(document);
  assert.ok(ui && ui.dlg.open, 'the dialog');
  assert.strictEqual(f3.posted, 0);
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  K.queues['/v1/config'].push(FULL);
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.ok(!ui.dlg.open);
  assert.deepStrictEqual(f3.requested, [btn]);
  assert.strictEqual(f3.posted, 1);
  assert.strictEqual(confirms, 1);
  // A form without requestSubmit (an old browser): submit().
  K.now += 61000;
  K.queues['/v1/config'].push(FULL);
  const f4 = form('post'); f4.requestSubmit = undefined; f4.submitted = 0; f4.submit = function () { this.submitted++; };
  dispatch(f4, btn); await flush();
  assert.strictEqual(f4.submitted, 1);
  // No answer to the check: the browser posts as it would have.
  K.now += 61000;
  K.queues['/v1/config'].push(['fail']);
  const f5 = form('post');
  dispatch(f5, btn); await flush();
  assert.strictEqual(f5.posted, 1);

  // The config poll turns minimal: the dialog opens; polls wait meanwhile; signed in, they go on.
  K.defaults['/v1/config'] = FULL;
  await advance(2000);   // a regular poll, full
  K.queues['/v1/config'].push(MINIMAL);
  const shown0 = K.shown;
  await advance(2000);
  assert.strictEqual(K.shown, shown0 + 1, 'opened by the poll');
  assert.ok(ui.dlg.open);
  const held = K.sent.length;
  await advance(4000);
  assert.strictEqual(K.sent.length, held, 'no poll goes out while the dialog is open');
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.ok(!ui.dlg.open);
  assert.ok(K.sent.length > held, 'polls go on after signing in');
  assert.deepStrictEqual(K.assigned, []);
  // Hidden while the answer turns minimal: no dialog until visible (the poll asks again then).
  document.hidden = true;
  K.queues['/v1/config'].push(MINIMAL);
  await advance(2000);
  // the timer had been set while visible: that poll ran and saw the minimal answer
  assert.ok(!ui.dlg.open, 'not while hidden');
  document.hidden = false;
  document.fire('visibilitychange');
  await flush();
  assert.ok(ui.dlg.open, 'opens once visible');
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.ok(!ui.dlg.open);

  // pageshow from the back-forward cache and `online` poll at once, one request at a time.
  await advance(2000);
  K.timers = K.timers.filter(() => true);
  let c0 = sentTo('/v1/config').length;
  await fireWin('pageshow', {persisted: false});
  assert.strictEqual(sentTo('/v1/config').length, c0, 'a normal load: nothing extra');
  await fireWin('pageshow', {persisted: true});
  assert.strictEqual(sentTo('/v1/config').length, c0 + 1, 'restored: one poll');
  c0 = sentTo('/v1/config').length;
  await fireWin('online');
  assert.strictEqual(sentTo('/v1/config').length, c0 + 1, 'online: one poll');
  K.queues['/v1/config'].push(['hang']);
  c0 = sentTo('/v1/config').length;
  (winL.online || []).forEach(f => f({}));
  (winL.online || []).forEach(f => f({}));
  await fireWin('pageshow', {persisted: true});
  assert.strictEqual(sentTo('/v1/config').length, c0 + 1, 'never two in flight');
  // That hung poll is aborted at 15 s; the loop backs off to 4 s.
  await advance(15000);
  const next = K.timers.filter(t => t.at > K.now).map(t => t.at - K.now);
  assert.ok(next.indexOf(4000) >= 0, 'backed off: ' + JSON.stringify(next));
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

/// `PrismDraft`: sessionStorage by path, and nothing breaks when storage is missing, refuses or
/// throws.
const DRAFT: &str = r#"
const src = fs.readFileSync(process.argv[2], 'utf8');
function draftWith(storage) {
  const sb = sandboxOf({pathname: '/train/review', search: '', hash: '', origin: 'https://h', assign() {}});
  if (storage !== 'throws') sb.sessionStorage = storage;
  const ctx = vm.createContext(sb);
  // A getter that throws (a browser refusing storage): defined inside the context, where it
  // throws (a getter on the sandbox object does not reach the script).
  if (storage === 'throws') vm.runInContext('Object.defineProperty(globalThis, "sessionStorage", {get() { throw new Error("SecurityError"); }})', ctx);
  assert.throws(() => { if (storage === 'throws') vm.runInContext('sessionStorage', ctx); else throw new Error('n/a'); });
  vm.runInContext(src, ctx);
  return vm.runInContext('PrismDraft', ctx);
}
const store = () => { const m = new Map(); return {m, getItem: k => m.has(k) ? m.get(k) : null, setItem: (k, v) => m.set(k, String(v)), removeItem: k => m.delete(k)}; };
// Working storage: a JSON round trip under a key of this path.
const s = store(), D = draftWith(s);
const value = {id: 'cam/a.jpg', rev: '3', labels: {boxes: [{label: 'person', _n: 0, _checked: true}]}, selected: 0};
assert.strictEqual(D.save('review', value), true);
assert.deepStrictEqual([...s.m.keys()], ['prism:draft:/train/review:review']);
assert.strictEqual(JSON.stringify(D.load('review')), JSON.stringify(value), 'a JSON round trip, `_` fields too');
assert.strictEqual(D.load('other'), null);
D.drop('review');
assert.strictEqual(D.load('review'), null);
s.m.set('prism:draft:/train/review:review', '{not json');
assert.strictEqual(D.load('review'), null, 'a damaged draft is none');
// Restore or drop.
assert.strictEqual(D.verdict(null, 'a', '1'), 'none');
assert.strictEqual(D.verdict('x', 'a', '1'), 'none');
assert.strictEqual(D.verdict({id: 'b', rev: '1'}, 'a', '1'), 'other');
assert.strictEqual(D.verdict({id: 'a', rev: '1'}, 'a', '1'), 'restore');
assert.strictEqual(D.verdict({id: 'a', rev: '1'}, 'a', '2'), 'drop');
assert.strictEqual(D.verdict({id: 'a', rev: ''}, 'a', ''), 'restore', 'never saved, still never saved');
assert.strictEqual(D.verdict({id: 'a', rev: ''}, 'a', 'r1'), 'drop', 'saved by someone since');
// No storage, a getter that throws, a full one, one that throws on read.
for (const st of [null, undefined, 'throws',
    {getItem() { throw new Error('x'); }, setItem() { throw new Error('QuotaExceededError'); }, removeItem() { throw new Error('x'); }}]) {
  const d = draftWith(st);
  assert.strictEqual(d.save('review', value), false);
  assert.strictEqual(d.load('review'), null);
  d.drop('review');
}
console.log('ok');
"#;

#[test]
fn polls_carry_the_poll_header_and_a_401_goes_to_the_login_page() {
    if have_node() {
        run(AUTH);
    }
}

#[test]
fn the_scheduler_never_overlaps_sleeps_while_hidden_and_backs_off() {
    if have_node() {
        run(SCHEDULER);
    }
}

#[test]
fn the_train_area_keeps_one_chain_per_list() {
    if have_node() {
        run(TRAIN_AREA);
    }
}

#[test]
fn a_401_opens_the_sign_in_dialog_and_repeats_the_requests_once_signed_in() {
    if have_node() {
        run(&format!("{KIT}{DIALOG}"));
    }
}

#[test]
fn without_a_dialog_a_401_goes_to_the_login_page_with_the_hash() {
    if have_node() {
        run(&format!("{KIT}{NO_DIALOG}"));
    }
}

#[test]
fn native_forms_check_the_session_and_the_poll_wakes_on_pageshow_and_online() {
    if have_node() {
        run(&format!("{KIT}{PAGE}"));
    }
}

#[test]
fn drafts_survive_missing_and_throwing_storage() {
    if have_node() {
        run(&format!("{KIT}{DRAFT}"));
    }
}

/// `PrismMemory` (live.js): the "Start anyway" checkbox beside a memory commit refusal shows only
/// with one, comes unticked when it appears, keeps a tick while the refusal stays, and is
/// disabled (so a native form post leaves it out) and unticked once it goes.
const MEMORY: &str = r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({console, Promise});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const M = vm.runInContext('PrismMemory', ctx);
function row() {
  const box = {type: 'checkbox', name: 'ignore_memory_check', checked: true, disabled: true};
  const words = {textContent: ''};
  return {hidden: true, box, words,
    querySelector: s => s.indexOf('checkbox') >= 0 ? box : s.indexOf('data-memory-label') >= 0 ? words : null};
}
const refusal = {refusal: 'this auto-label run needs about 10.0 GB ...', override_label: 'Start anyway: this job may grow the paging file into the last 16 GB of C:'};
const r = row();
M.sync(r, null);
assert.deepStrictEqual([r.hidden, r.box.disabled, r.box.checked, r.words.textContent], [true, true, false, ''], 'no refusal: hidden');
assert.strictEqual(M.ticked(r), false);
r.box.checked = true;   // a restored form ticks a hidden box: it is still not "ticked"
assert.strictEqual(M.ticked(r), false, 'a hidden box never counts');
M.sync(r, refusal);
assert.deepStrictEqual([r.hidden, r.box.disabled, r.box.checked], [false, false, false], 'appears unticked');
assert.strictEqual(r.words.textContent, refusal.override_label);
assert.strictEqual(M.ticked(r), false);
r.box.checked = true;
M.sync(r, refusal);
assert.strictEqual(M.ticked(r), true, 'the tick stays while the refusal does');
M.sync(r, {});
assert.deepStrictEqual([r.hidden, r.box.disabled, r.box.checked], [true, true, false], 'a preview without one hides it');
M.sync(null, refusal);   // a page without the row
console.log('ok');
"#;

#[test]
fn the_start_anyway_box_shows_only_beside_a_memory_refusal() {
    if have_node() {
        run(MEMORY);
    }
}

/// `PrismLive.ui` (how a polled server value reaches a form control: an unedited one takes it, an
/// edited one keeps the user's value and says when the server changed it too) and `PrismLive.keep`
/// / `patch` (when a polled re-render is skipped, deferred while the user is in the section, or
/// applied per keyed child), on fake controls and nodes. Pins what the Config page relies on.
const UI: &str = r#"
const location = {pathname: '/config', search: '', hash: '', origin: 'https://h',
  assign(u) { K.assigned.push(u); }, reload() {}, replace() {}};
const sandbox = sandboxOf(location);
const document = makeDocument();
function Option(text, value, defaultSelected, selected) {
  this.text = text; this.value = value; this.defaultSelected = defaultSelected; this.selected = selected;
}
Object.assign(sandbox, {document, Option, history: {}, scrollX: 0, scrollY: 0, scrollTo() {}, addEventListener() {}});
sandbox.window = sandbox;
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const L = vm.runInContext('PrismLive', ctx), U = L.ui;

function classes() {
  const s = new Set(), adds = [];
  return {s, adds, add(c) { s.add(c); adds.push(c); }, remove(c) { s.delete(c); }, contains(c) { return s.has(c); },
    toggle(c, on) { if (on === undefined) on = !s.has(c); if (on) s.add(c); else s.delete(c); return on; }};
}
// The <label> (or <td>) a control sits in: where its "Changed elsewhere" note goes.
function holder() {
  return {children: [],
    querySelector(sel) { return sel === '.live-note' ? this.children.find(c => c.className === 'live-note') || null : null; },
    appendChild(c) { c.parentNode = this; c.remove = () => { this.children = this.children.filter(x => x !== c); }; this.children.push(c); return c; }};
}
function control(props, inLabel) {
  const h = holder();
  return Object.assign({tagName: 'INPUT', classList: classes(), parentNode: h, offsetWidth: 0,
    closest: (sel) => (inLabel === false ? null : (sel === 'label, td' ? h : null))}, props);
}
const field = (type, v, inLabel) => control({type, value: String(v), defaultValue: String(v)}, inLabel);
const checkbox = (on) => control({type: 'checkbox', checked: on, defaultChecked: on});
function select(values, chosen) {
  const s = control({type: 'select-one', tagName: 'SELECT',
    options: values.map(v => ({value: v, selected: v === chosen, defaultSelected: v === chosen})),
    add(o) { this.options.push(o); }});
  // (a getter Object.assign would have read once)
  Object.defineProperty(s, 'value', {get() { const o = this.options.find(x => x.selected); return o ? o.value : ''; }});
  return s;
}
const noteOf = (e) => { const n = e.parentNode.querySelector('.live-note'); return n ? n.textContent : null; };
const notes = (e) => e.parentNode.children.length;
const flashes = (e) => e.classList.adds.filter(c => c === 'flash').length;
const has = (e, c) => e.classList.contains(c);
const ELSEWHERE = (v) => 'Changed elsewhere to ' + v + '; saving replaces it with your value.';

// edited(): a control differs from what the server last showed (its default state).
{
  const t = field('text', 'a');
  assert.strictEqual(U.edited(t), false);
  t.value = 'b'; assert.strictEqual(U.edited(t), true);
  const n = field('number', '0.50');
  n.value = '0.5'; assert.strictEqual(U.edited(n), false, 'the same number, spelt otherwise');
  n.value = ' 0.5 '; assert.strictEqual(U.edited(n), false, 'spaces around it');
  n.value = '0.6'; assert.strictEqual(U.edited(n), true);
  n.value = ''; assert.strictEqual(U.edited(n), true, 'emptied');
  const n0 = field('number', '');
  assert.strictEqual(U.edited(n0), false);
  n0.value = '0'; assert.strictEqual(U.edited(n0), true, 'filled in');
  const c = checkbox(false);
  assert.strictEqual(U.edited(c), false);
  c.checked = true; assert.strictEqual(U.edited(c), true);
  const s = select(['a', 'b'], 'a');
  assert.strictEqual(U.edited(s), false);
  s.options[0].selected = false; s.options[1].selected = true;
  assert.strictEqual(U.edited(s), true);
}
// current / baseline / same.
{
  const c = checkbox(true);
  assert.strictEqual(U.current(c), true);
  c.checked = false;
  assert.strictEqual(U.current(c), false);
  assert.strictEqual(U.baseline(c), true);
  const t = field('text', 'a'); t.value = 'b';
  assert.strictEqual(U.current(t), 'b');
  assert.strictEqual(U.baseline(t), 'a');
  assert.strictEqual(U.baseline(select(['a', 'b'], 'b')), 'b');
  assert.strictEqual(U.baseline(select(['a', 'b'], null)), 'a', 'no default: the first option');
  assert.strictEqual(U.baseline(select([], null)), '', 'no options');
  const n = field('number', '1');
  assert.strictEqual(U.same(n, '0.50', 0.5), true, 'numbers by value');
  assert.strictEqual(U.same(n, '', 0), false, 'empty is not 0');
  assert.strictEqual(U.same(n, '', null), true, 'empty is null');
  assert.strictEqual(U.same(t, '0.50', 0.5), false, 'text by its letters');
  assert.strictEqual(U.same(t, null, ''), true);
  assert.strictEqual(U.same(t, undefined, ''), true);
  assert.strictEqual(U.same(t, 3, '3'), true);
  assert.strictEqual(U.same(c, 1, true), true, 'a checkbox by truth');
  assert.strictEqual(U.same(c, '', true), false);
  assert.strictEqual(U.same(c, 0, false), true);
}
// set(): value and default together; a select takes an unknown value as one extra option.
{
  const t = field('text', 'a');
  U.set(t, 'b'); assert.deepStrictEqual([t.value, t.defaultValue], ['b', 'b']);
  U.set(t, null); assert.deepStrictEqual([t.value, t.defaultValue], ['', '']);
  U.set(t, 3); assert.deepStrictEqual([t.value, t.defaultValue], ['3', '3']);
  const c = checkbox(false);
  U.set(c, 1); assert.deepStrictEqual([c.checked, c.defaultChecked], [true, true]);
  U.set(c, ''); assert.deepStrictEqual([c.checked, c.defaultChecked], [false, false]);
  const s = select(['a', 'b'], 'a');
  const state = () => s.options.map(o => [o.value, o.selected, o.defaultSelected]);
  U.set(s, 'b');
  assert.deepStrictEqual(state(), [['a', false, false], ['b', true, true]]);
  U.set(s, 'zzz');
  assert.deepStrictEqual(state(), [['a', false, false], ['b', false, false], ['zzz', true, true]]);
  assert.strictEqual(s.options[2].text, 'zzz — not a known option');
  U.set(s, 'zzz');
  assert.strictEqual(s.options.length, 3, 'the same unknown value again: no second option');
  U.set(s, 'yyy');
  assert.deepStrictEqual(state(), [['a', false, false], ['b', false, false], ['zzz', false, false], ['yyy', true, true]],
    'another unknown value: one more option, the first one stays');
  U.set(s, 'a');
  assert.strictEqual(s.options.length, 4);
  assert.strictEqual(U.edited(s), false);
}
// refresh(): an unedited control takes the server value (a flash only when it changed).
{
  const t = field('text', 'a');
  assert.strictEqual(U.refresh(t, 'a'), true);
  assert.strictEqual(flashes(t), 0, 'unchanged: no flash');
  assert.strictEqual(U.refresh(t, 'b'), true);
  assert.deepStrictEqual([t.value, t.defaultValue, flashes(t)], ['b', 'b', 1]);
  assert.strictEqual(notes(t), 0);
  const n = field('number', '0.50');
  assert.strictEqual(U.refresh(n, 0.5), true);
  assert.strictEqual(flashes(n), 0, '0.50 and 0.5 are the same');
  assert.strictEqual(n.value, '0.5', 'the server spelling is shown');
  const c = checkbox(false);
  assert.strictEqual(U.refresh(c, true), true);
  assert.deepStrictEqual([c.checked, c.defaultChecked, flashes(c)], [true, true, 1]);
  const s = select(['a', 'b'], 'a');
  assert.strictEqual(U.refresh(s, 'q'), true);
  assert.deepStrictEqual([s.options.length, s.value, flashes(s)], [3, 'q', 1]);
  assert.strictEqual(U.refresh(s, 'q'), true);
  assert.deepStrictEqual([s.options.length, flashes(s)], [3, 1], 'a repeat adds nothing and does not flash');
}
// refresh(): an edited control keeps the user's value; the server changing it too is noted.
{
  const t = field('text', 'a');
  t.value = 'mine'; t.classList.add('edited');
  assert.strictEqual(U.refresh(t, 'theirs'), false);
  assert.deepStrictEqual([t.value, t.defaultValue], ['mine', 'a'], 'the user\'s value and the old default stay');
  assert.strictEqual(noteOf(t), ELSEWHERE('theirs'));
  assert.ok(has(t, 'edited-conflict') && has(t, 'edited'));
  assert.strictEqual(U.refresh(t, 'x', v => '«' + v + '»'), false);
  assert.strictEqual(noteOf(t), ELSEWHERE('«x»'), 'the page describes the value');
  assert.strictEqual(notes(t), 1, 'one note, rewritten');
  // The server is back at what the page showed: the note goes, the edit stays.
  assert.strictEqual(U.refresh(t, 'a'), false);
  assert.strictEqual(noteOf(t), null);
  assert.strictEqual(notes(t), 0);
  assert.ok(!has(t, 'edited-conflict') && has(t, 'edited'));
  assert.strictEqual(t.value, 'mine');
  // The server changes to the user's value: no longer an edit.
  U.refresh(t, 'zzz');
  assert.strictEqual(noteOf(t), ELSEWHERE('zzz'));
  assert.strictEqual(U.refresh(t, 'mine'), true);
  assert.deepStrictEqual([t.value, t.defaultValue], ['mine', 'mine']);
  assert.strictEqual(noteOf(t), null);
  assert.ok(!has(t, 'edited-conflict') && !has(t, 'edited'));
  assert.strictEqual(flashes(t), 0, 'no flash for the user\'s own value');
  // An edited control the server did not change: no note.
  const u = field('text', 'a'); u.value = 'b';
  assert.strictEqual(U.refresh(u, 'a'), false);
  assert.strictEqual(notes(u), 0);
  // An edited number: the server's 0.5 is its 0.50, the user's 0.7 stays without a note.
  const n = field('number', '0.50'); n.value = '0.7';
  assert.strictEqual(U.refresh(n, 0.5), false);
  assert.strictEqual(notes(n), 0);
  assert.strictEqual(n.value, '0.7');
  // Outside a <label> / <td> the note goes in the parent.
  const p = field('text', 'a', false); p.value = 'b';
  U.refresh(p, 'c');
  assert.strictEqual(noteOf(p), ELSEWHERE('c'));
  // An edited checkbox the server sets to the user's state.
  const c = checkbox(false); c.checked = true;
  assert.strictEqual(U.refresh(c, true), true);
  assert.strictEqual(c.defaultChecked, true);
}
// markEdits(): input and change mark a named control `edited` while it differs.
{
  const form = el('form');
  U.markEdits(form);
  const t = field('text', 'a'); t.name = 'x';
  t.value = 'b'; form.listeners.input.forEach(h => h({target: t}));
  assert.ok(has(t, 'edited'));
  t.value = 'a'; form.listeners.change.forEach(h => h({target: t}));
  assert.ok(!has(t, 'edited'));
  const anon = field('text', 'a'); anon.value = 'b';
  form.listeners.input.forEach(h => h({target: anon}));
  assert.ok(!has(anon, 'edited'), 'unnamed controls are left alone');
}

// keep() / patch(): nodes with children, siblings and `contains`; `marked` = a chart under the mouse.
class N {
  constructor(key) { this.tagName = 'DIV'; this.children = []; this.parentNode = null; this.dataset = {}; this.marked = false;
    if (key !== undefined) this.dataset.patchKey = key; }
  get firstChild() { return this.children[0] || null; }
  get nextSibling() { const p = this.parentNode; return p ? p.children[p.children.indexOf(this) + 1] || null : null; }
  detach() { const p = this.parentNode; if (p) { p.children.splice(p.children.indexOf(this), 1); this.parentNode = null; } }
  insertBefore(n, ref) {
    n.detach();
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(n); else this.children.splice(i, 0, n);
    n.parentNode = this; return n;
  }
  appendChild(n) { return this.insertBefore(n, null); }
  replaceWith(n) { const p = this.parentNode; n.detach(); p.children[p.children.indexOf(this)] = n; n.parentNode = p; this.parentNode = null; }
  remove() { this.detach(); }
  contains(x) { for (let e = x; e; e = e.parentNode) if (e === this) return true; return false; }
  querySelectorAll() { return []; }
  querySelector(sel) { return this.marked && sel.indexOf('[data-pointer-in]') >= 0 ? new N() : null; }
}
const press = (t) => document.listeners.pointerdown.forEach(f => f({target: t}));
const lift = () => document.listeners.pointerup.forEach(f => f({}));
{
  let renders = 0;
  const render = () => { renders++; };
  assert.strictEqual(L.keep(null, 's', render), false, 'no section');
  const root = new N(), inner = root.appendChild(new N());
  assert.strictEqual(L.keep(root, 's1', render), true);
  assert.deepStrictEqual([renders, root._keepSig], [1, 's1']);
  assert.strictEqual(L.keep(root, 's1', render), false, 'the same data: skipped');
  assert.strictEqual(renders, 1);
  // The mouse is pressed inside: deferred, the new signature not taken.
  press(inner);
  assert.strictEqual(L.keep(root, 's2', render), false);
  assert.deepStrictEqual([renders, root._keepSig], [1, 's1']);
  lift();
  assert.strictEqual(L.keep(root, 's2', render), true, 'released: the deferred render runs');
  assert.deepStrictEqual([renders, root._keepSig], [2, 's2']);
  // Pressed elsewhere: not busy here.
  press(new N());
  assert.strictEqual(L.keep(root, 's3', render), true);
  lift();
  // A chart under the mouse (data-pointer-in) defers too.
  root.marked = true;
  assert.strictEqual(L.keep(root, 's4', render), false);
  root.marked = false;
  assert.strictEqual(L.keep(root, 's4', render), true);
  // No signature: always rendered.
  assert.strictEqual(L.keep(root, undefined, render), true);
  assert.strictEqual(L.keep(root, undefined, render), true);
  assert.strictEqual(renders, 6);
  // The first render of a section is never deferred (nothing is shown yet).
  const fresh = new N(); fresh.marked = true;
  assert.strictEqual(L.keep(fresh, 'a', render), true);
}
{
  const box = new N();
  let builds = [];
  const item = (key, sig) => ({key, sig, build: () => { builds.push(key); return new N(); }});
  const keys = () => box.children.map(c => c.dataset.patchKey);
  L.patch(box, [item('a', 1), item('b', 1), item('c', 1)]);
  assert.deepStrictEqual(keys(), ['a', 'b', 'c']);
  assert.deepStrictEqual(builds, ['a', 'b', 'c']);
  assert.deepStrictEqual(box.children.map(c => c._keepSig), [1, 1, 1]);
  const [a, b, c] = box.children;
  // Reordered, unchanged: moved, never rebuilt.
  builds = [];
  L.patch(box, [item('c', 1), item('a', 1), item('b', 1)]);
  assert.deepStrictEqual(keys(), ['c', 'a', 'b']);
  assert.deepStrictEqual(builds, []);
  assert.ok(box.children[0] === c && box.children[1] === a && box.children[2] === b, 'the same nodes');
  // A changed signature: that child is rebuilt in its place.
  L.patch(box, [item('c', 1), item('a', 2), item('b', 1)]);
  assert.deepStrictEqual(builds, ['a']);
  assert.deepStrictEqual(keys(), ['c', 'a', 'b']);
  assert.ok(box.children[1] !== a && a.parentNode === null, 'the old one is out');
  assert.strictEqual(box.children[1]._keepSig, 2);
  assert.ok(box.children[0] === c && box.children[2] === b);
  // A key gone is removed; a new one is built where it belongs.
  builds = [];
  L.patch(box, [item('d', 1), item('c', 1), item('b', 1)]);
  assert.deepStrictEqual(keys(), ['d', 'c', 'b']);
  assert.deepStrictEqual(builds, ['d']);
  // The mouse pressed in a child: its change waits (signature not taken), the rest goes on.
  press(b.appendChild(new N()));
  builds = [];
  L.patch(box, [item('d', 2), item('c', 1), item('b', 2)]);
  assert.deepStrictEqual(builds, ['d']);
  assert.ok(box.children[2] === b && b._keepSig === 1);
  lift();
  L.patch(box, [item('d', 2), item('c', 1), item('b', 2)]);
  assert.deepStrictEqual(builds, ['d', 'b'], 'released: rebuilt on the next patch');
  assert.deepStrictEqual(keys(), ['d', 'c', 'b']);
  // Nothing left.
  L.patch(box, []);
  assert.deepStrictEqual(keys(), []);
}
console.log('ok');
"#;

#[test]
fn polled_values_refresh_unedited_fields_and_keep_and_patch_skip_or_defer_re_renders() {
    if have_node() {
        run(&format!("{KIT}{UI}"));
    }
}

/// The config poll of `live.js` (GET /v1/config behind every page): the first poll at load, then
/// one every 2 s with `If-None-Match`; failures back off 4, 8, 16, 30 s and say so in the header,
/// a success clears that; nothing while the tab is hidden, a poll at once when it shows, on
/// focus, on `pageshow` from the back-forward cache and on `online`; `PrismLive.poll(true)`
/// skips the ETag, shares a request in flight and resolves to the last payload when it fails; a
/// signed-out (minimal) answer keeps the last payload and opens the sign-in dialog, and once
/// signed in a fresh full answer is fetched and the 2 s period goes on.
const CONFIG_POLL: &str = r#"
const location = {pathname: '/config', search: '', hash: '', origin: 'https://h',
  assign(u) { K.assigned.push(u); }, reload() {}, replace() {}};
const sandbox = sandboxOf(location);
const document = makeDocument();
const status = el('span'); status.hidden = true;
document.getElementById = (id) => id === 'live-status' ? status : null;
const winL = {};
Object.assign(sandbox, {document, HTMLDialogElement: function () {}, history: {}, scrollX: 0, scrollY: 0,
  scrollTo() {}, confirm: () => true,
  addEventListener(t, f) { (winL[t] = winL[t] || []).push(f); }});
sandbox.window = sandbox;
const fireWin = async (t, ev) => { (winL[t] || []).forEach(f => f(ev || {})); await flush(); };
const FULL = (rev) => [200, {config: {rev}, revision: rev, epoch: 'e', pendingCount: 0}, {ETag: '"r' + rev + '"'}];
const MINIMAL = [200, {success: true, revision: 1, epoch: 'e', pendingCount: 0}];
const URL = '/v1/config';
K.queues[URL] = [FULL(1)];
const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const L = vm.runInContext('PrismLive', ctx);
const seen = [];
L.onConfig((j) => seen.push(j.revision));
const polls = () => sentTo(URL).length;
const lastSent = () => sentTo(URL).slice(-1)[0];
const etagOf = (s) => s.init.headers['If-None-Match'];
// Timers still to run, in ms from now (a request in flight has its 15 s abort timer).
const pending = () => K.timers.map(t => t.at - K.now).sort((a, b) => a - b);
const line = () => status.hidden ? null : status.textContent;

(async () => {
  await flush();
  assert.strictEqual(polls(), 1, 'the first poll at load');
  assert.strictEqual(pollMark(lastSent()), '1', 'a poll');
  assert.strictEqual(etagOf(lastSent()), undefined, 'no ETag yet');
  assert.strictEqual(lastSent().init.cache, 'no-store');
  assert.deepStrictEqual(seen, [1]);
  assert.deepStrictEqual(pending(), [2000], 'the next one 2 s on');
  // 304: nothing new for the pages.
  K.defaults[URL] = [304];
  await advance(2000);
  assert.strictEqual(polls(), 2);
  assert.strictEqual(etagOf(lastSent()), '"r1"');
  assert.deepStrictEqual(seen, [1]);
  assert.deepStrictEqual(pending(), [2000]);

  // No answer: the period backs off and the header says when it tries again.
  K.defaults[URL] = ['fail'];
  const backoff = [];
  for (const wait of [2000, 4000, 8000, 16000, 30000]) {
    const n = polls();
    await advance(wait);
    assert.strictEqual(polls(), n + 1, 'one poll after ' + wait + ' ms');
    backoff.push([pending()[0], line()]);
  }
  assert.deepStrictEqual(backoff, [
    [4000, 'Live updates: server not answering; retrying in 4 s'],
    [8000, 'Live updates: server not answering; retrying in 8 s'],
    [16000, 'Live updates: server not answering; retrying in 16 s'],
    [30000, 'Live updates: server not answering; retrying in 30 s'],
    [30000, 'Live updates: server not answering; retrying in 30 s']]);
  K.queues[URL].push([500, {}]);
  await advance(30000);
  assert.strictEqual(line(), 'Live updates: server answered HTTP 500; retrying in 30 s');
  assert.strictEqual(L.last().revision, 1, 'the last payload stays');
  // Answering again: the line goes, the period is 2 s again.
  K.defaults[URL] = FULL(2);
  await advance(30000);
  assert.strictEqual(line(), null);
  assert.strictEqual(status.textContent, '');
  assert.deepStrictEqual(seen, [1, 2]);
  assert.deepStrictEqual(pending(), [2000]);

  // Hidden: no timer and no request, however long; shown again: a poll at once.
  K.defaults[URL] = [304];
  document.hidden = true; document.fire('visibilitychange'); await flush();
  assert.deepStrictEqual(pending(), [], 'no timer while hidden');
  let n = polls();
  await advance(120000);
  await fireWin('focus');
  await fireWin('pageshow', {persisted: true});
  await fireWin('online');
  assert.strictEqual(polls(), n, 'nothing while hidden');
  document.hidden = false; document.fire('visibilitychange'); await flush();
  assert.strictEqual(polls(), n + 1, 'a poll when shown');
  assert.deepStrictEqual(pending(), [2000]);
  // Focus, back from the back-forward cache, the network back: a poll at once each.
  n = polls();
  await fireWin('focus');
  assert.strictEqual(polls(), n + 1, 'focus');
  await fireWin('pageshow', {persisted: false});
  assert.strictEqual(polls(), n + 1, 'a normal pageshow: nothing');
  await fireWin('pageshow', {persisted: true});
  assert.strictEqual(polls(), n + 2, 'pageshow from the cache');
  await fireWin('online');
  assert.strictEqual(polls(), n + 3, 'online');

  // PrismLive.poll(true): no ETag; two callers and the timer share one request in flight.
  await advance(2000);
  K.queues[URL].push(['hang']);
  n = polls();
  let a = null, b = null;
  L.poll(true).then(j => { a = j; });
  L.poll(false).then(j => { b = j; });
  await flush();
  assert.strictEqual(polls(), n + 1, 'one request for both');
  assert.strictEqual(etagOf(lastSent()), undefined, 'forced: no ETag');
  await advance(2000);
  assert.strictEqual(polls(), n + 1, 'the timer joins it');
  answer(lastSent(), ...FULL(3));
  await flush();
  assert.ok(a && a.revision === 3 && b && b.revision === 3, 'both get the payload');
  assert.deepStrictEqual(seen, [1, 2, 3]);
  assert.deepStrictEqual(pending(), [2000], 'one timer after it');
  // A forced poll that fails resolves to the last payload (config.js reads it).
  K.queues[URL].push(['fail']);
  const got = await L.poll(true);
  assert.strictEqual(got.revision, 3);
  await advance(2000);
  assert.strictEqual(line(), null, 'the next poll answered');

  // Signed out (the answer turns minimal): the dialog opens and the last payload stays.
  K.queues[URL].push(MINIMAL);
  const shown0 = K.shown;
  await advance(2000);
  assert.strictEqual(K.shown, shown0 + 1, 'the sign-in dialog');
  const ui = dialogOf(document);
  assert.ok(ui && ui.dlg.open);
  assert.strictEqual(L.last().revision, 3);
  assert.deepStrictEqual(seen, [1, 2, 3], 'nothing emitted');
  assert.strictEqual(line(), null);
  // Signed in before the next poll: one fresh full answer (no ETag), then every 2 s again.
  n = polls();
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  K.queues[URL].push(FULL(4));
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.ok(!ui.dlg.open);
  assert.strictEqual(polls(), n + 1, 'one poll after signing in');
  assert.strictEqual(etagOf(lastSent()), undefined, 'a fresh answer');
  assert.deepStrictEqual(seen, [1, 2, 3, 4]);
  assert.deepStrictEqual(pending(), [2000], 'the loop goes on');
  // Signed out again, the next poll held behind the dialog: signing in sends it, and only it.
  K.queues[URL].push(MINIMAL);
  await advance(2000);
  assert.ok(ui.dlg.open);
  n = polls();
  await advance(10000);
  assert.strictEqual(polls(), n, 'held while the dialog is open');
  K.queues['/login'] = [[200, {success: true, next: '/'}]];
  K.queues[URL].push(FULL(5));
  ui.pass.value = 'right'; submitDialog(ui); await flush();
  assert.strictEqual(polls(), n + 1, 'the held poll, once');
  assert.deepStrictEqual(seen, [1, 2, 3, 4, 5]);
  assert.deepStrictEqual(pending(), [2000]);
  assert.deepStrictEqual(K.assigned, [], 'never the login page');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"#;

#[test]
fn the_config_poll_backs_off_sleeps_while_hidden_wakes_and_signs_in_again() {
    if have_node() {
        run(&format!("{KIT}{CONFIG_POLL}"));
    }
}
