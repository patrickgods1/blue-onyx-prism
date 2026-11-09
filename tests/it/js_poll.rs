//! The pages' poll loops (`PrismPoll` in `assets/live.js`) and the Train area's two chains
//! (`assets/train-area.js`), run in `node` with fake timers, a fake `fetch` and a stub DOM.
//!
//! What they pin: a loop never has two requests in flight, never ticks while the tab is hidden,
//! runs at once when it shows again, backs off after errors, and a refresh of the Train area asks
//! the running chain instead of starting another (the old code started a second 3 s chain from the
//! 30 s timer and the `train-area:refresh` event, doubling the requests for good). Signed-in
//! requests: polls carry `X-Prism-Poll: 1` (so they do not keep a session alive) and a 401 sends
//! the page to the login page once, with the way back, instead of backing off.
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

/// Run `driver_js` with the asset paths as arguments; it asserts and exits non-zero on a failure.
fn run(driver_js: &str) {
    let dir = std::env::temp_dir().join(format!("bop-js-poll-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let driver = dir.join("driver.js");
    std::fs::write(&driver, driver_js).unwrap();
    let out = Command::new("node")
        .arg(&driver)
        .arg(asset("live.js"))
        .arg(asset("train-area.js"))
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
const sandbox = {
  console, Promise, Headers, location, setTimeout, clearTimeout,
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
  // A 401 of this server: to the login page with the way back, once; the request never
  // settles, so a poll loop does not back off or report an error meanwhile.
  replies.push(401, 401);
  const errors = [];
  const looped = Poll.create({run: () => fetch('/v1/train/summary'), period: 1000, env,
    onError: (n) => errors.push(n)});
  looped.start();
  assert.strictEqual(await settled(fetch('/v1/config')), false, 'a 401 never settles');
  await tick();
  assert.deepStrictEqual(assigned, ['/login?next=%2Fconfig%3Fx%3D1']);
  assert.deepStrictEqual(errors, [], 'no back-off');
  assert.strictEqual(looped.state().busy, true, 'the loop waits for the page to go');
  // On the login page itself nothing loops back to it.
  const ctx2 = vm.createContext({console, Promise, Headers,
    location: {pathname: '/login', search: '', origin: 'https://h', assign(u) { assigned.push(u); }},
    fetch: () => Promise.resolve({status: 401})});
  vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx2);
  vm.runInContext('fetch', ctx2)('/v1/x');
  await tick();
  assert.strictEqual(assigned.length, 1);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
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
