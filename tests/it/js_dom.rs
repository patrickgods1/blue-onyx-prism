//! Node tests of `assets/dom.js` (`PrismDom`, `PrismFetch`; QA-141): `PrismFetch.json` and
//! `postJson` against fake `fetch` answers (200 JSON, 200 `success:false`, 409, 500 HTML, 200 that
//! is not JSON, a rejected fetch; the Error's message, `status`, `body`, `conflict`; what a POST
//! sends) and `PrismDom.el` / `clear` / `report` on a tiny fake document.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).
use crate::js_harness::{asset, have_node, run_node_ok};

/// Load `assets/dom.js` in a context with a fake `document` and a scriptable `fetch`, then run
/// `body`; it asserts and exits non-zero on a failure.
fn run(body: &str) {
    let driver_js = format!(
        r#"
const assert = require('assert');
const fs = require('fs'), vm = require('vm');
class N {{
  constructor(tag) {{ this.tagName = tag; this.className = ''; this.children = []; this._text = ''; }}
  set textContent(t) {{ this._text = String(t); this.children = []; }}
  get textContent() {{ return this.children.length ? this.children.map(c => c.textContent).join('|') : this._text; }}
  get firstChild() {{ return this.children[0] || null; }}
  appendChild(c) {{ this.children.push(c); return c; }}
  removeChild(c) {{ this.children.splice(this.children.indexOf(c), 1); return c; }}
}}
const sent = [];
let answer = null;   // set by each case: {{status, text}} or {{reject: msg}}
const fetchStub = (url, init) => {{
  sent.push({{url, init}});
  if (answer.reject) return Promise.reject(new Error(answer.reject));
  return Promise.resolve({{
    ok: answer.status >= 200 && answer.status < 300,
    status: answer.status,
    json: () => new Promise((res, rej) => {{ try {{ res(JSON.parse(answer.text)); }} catch (e) {{ rej(e); }} }})
  }});
}};
const ctx = vm.createContext({{document: {{createElement: (t) => new N(t)}}, fetch: fetchStub}});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), ctx);
const D = vm.runInContext('PrismDom', ctx), F = vm.runInContext('PrismFetch', ctx);
const failure = (p) => p.then(() => assert.fail('resolved'), (e) => e);
{body}
"#
    );
    run_node_ok(&driver_js, &[asset("dom.js").as_os_str()]);
}

#[test]
fn json_resolves_with_the_parsed_answer() {
    if !have_node() {
        return;
    }
    run(r#"
(async () => {
  answer = {status: 200, text: '{"success":true,"n":3}'};
  const j = await F.json('/v1/x', {headers: {Accept: 'application/json'}});
  assert.strictEqual(j.n, 3);
  assert.strictEqual(sent[0].url, '/v1/x');
  assert.strictEqual(sent[0].init.headers.Accept, 'application/json');
  // An answer without `success` is a success.
  answer = {status: 200, text: '{"items":[]}'};
  assert.deepStrictEqual(Array.from((await F.json('/v1/y')).items), []);
  console.log('ok');
})().catch((e) => { console.error(e); process.exit(1); });
"#);
}

#[test]
fn json_rejects_with_message_status_body_and_conflict() {
    if !have_node() {
        return;
    }
    run(r#"
(async () => {
  // success:false on a 200: the server's message; the body is kept.
  answer = {status: 200, text: '{"success":false,"message":"no such frame"}'};
  let e = await failure(F.json('/v1/x'));
  assert.strictEqual(e.message, 'no such frame');
  assert.strictEqual(e.status, 200);
  assert.strictEqual(e.body.success, false);
  assert.strictEqual(e.conflict, false);
  // 409 is a conflict, with or without a body message.
  answer = {status: 409, text: '{"success":false,"message":"changed meanwhile"}'};
  e = await failure(F.json('/v1/x'));
  assert.strictEqual(e.status, 409);
  assert.strictEqual(e.conflict, true);
  assert.strictEqual(e.message, 'changed meanwhile');
  // `conflict` in the body counts on any status.
  answer = {status: 400, text: '{"success":false,"conflict":true}'};
  e = await failure(F.json('/v1/x'));
  assert.strictEqual(e.conflict, true);
  assert.strictEqual(e.message, 'HTTP 400');
  // A not-ok answer with an ok-looking body still rejects.
  answer = {status: 503, text: '{"success":true}'};
  e = await failure(F.json('/v1/x'));
  assert.strictEqual(e.message, 'HTTP 503');
  assert.strictEqual(e.status, 503);
  console.log('ok');
})().catch((e) => { console.error(e); process.exit(1); });
"#);
}

#[test]
fn json_names_the_status_when_the_answer_is_not_json() {
    if !have_node() {
        return;
    }
    run(r#"
(async () => {
  answer = {status: 500, text: '<html>Internal Server Error</html>'};
  let e = await failure(F.json('/v1/x'));
  assert.strictEqual(e.message, 'HTTP 500');
  assert.strictEqual(e.status, 500);
  assert.strictEqual(e.body, null);
  assert.strictEqual(e.conflict, false);
  // A 200 that is not JSON (a proxy's page) is no success either.
  answer = {status: 200, text: '<html>login</html>'};
  e = await failure(F.json('/v1/x'));
  assert.strictEqual(e.message, 'HTTP 200');
  assert.strictEqual(e.body, null);
  // So is a JSON `null`.
  answer = {status: 200, text: 'null'};
  e = await failure(F.json('/v1/x'));
  assert.strictEqual(e.message, 'HTTP 200');
  console.log('ok');
})().catch((e) => { console.error(e); process.exit(1); });
"#);
}

#[test]
fn json_passes_a_rejected_fetch_through_untouched() {
    if !have_node() {
        return;
    }
    run(r#"
(async () => {
  answer = {reject: 'Failed to fetch'};
  const e = await failure(F.json('/v1/x'));
  assert.strictEqual(e.message, 'Failed to fetch');
  assert.strictEqual(e.status, undefined);
  console.log('ok');
})().catch((e) => { console.error(e); process.exit(1); });
"#);
}

#[test]
fn post_json_sends_the_body_as_json() {
    if !have_node() {
        return;
    }
    run(r#"
(async () => {
  answer = {status: 200, text: '{"success":true,"count":2}'};
  const j = await F.postJson('/v1/review/accept-auto', {dry_run: true, camera: 'front'});
  assert.strictEqual(j.count, 2);
  const s = sent[0];
  assert.strictEqual(s.url, '/v1/review/accept-auto');
  assert.strictEqual(s.init.method, 'POST');
  assert.strictEqual(s.init.headers['Content-Type'], 'application/json');
  assert.strictEqual(s.init.headers.Accept, 'application/json');
  assert.strictEqual(s.init.body, '{"dry_run":true,"camera":"front"}');
  answer = {status: 409, text: '{"success":false,"message":"busy"}'};
  const e = await failure(F.postJson('/v1/x', {}));
  assert.strictEqual(e.conflict, true);
  console.log('ok');
})().catch((e) => { console.error(e); process.exit(1); });
"#);
}

#[test]
fn el_sets_class_and_text_only_when_given() {
    if !have_node() {
        return;
    }
    run(r#"
let e = D.el('div');
assert.strictEqual(e.tagName, 'div');
assert.strictEqual(e.className, '');
assert.strictEqual(e.textContent, '');
e = D.el('p', 'note', 'hello');
assert.strictEqual(e.className, 'note');
assert.strictEqual(e.textContent, 'hello');
// null and undefined text set nothing; "" and 0 are text.
assert.strictEqual(D.el('p', null, null).textContent, '');
assert.strictEqual(D.el('p', null, undefined).textContent, '');
assert.strictEqual(D.el('p', null, 0).textContent, '0');
assert.strictEqual(D.el('p', '', '').textContent, '');
console.log('ok');
"#);
}

#[test]
fn clear_removes_every_child() {
    if !have_node() {
        return;
    }
    run(r#"
const box = D.el('div');
box.appendChild(D.el('i', null, 'a'));
box.appendChild(D.el('i', null, 'b'));
D.clear(box);
assert.strictEqual(box.children.length, 0);
D.clear(box);
assert.strictEqual(box.children.length, 0);
console.log('ok');
"#);
}

#[test]
fn report_prints_lines_then_a_capped_warning_list() {
    if !have_node() {
        return;
    }
    run(r#"
const box = D.el('div', null, 'old text');
D.report(box, ['one', 'two']);
assert.deepStrictEqual(box.children.map(c => c.tagName + ':' + c.textContent), ['div:one', 'div:two']);
// Warnings follow in a `warn-list`; an empty list adds nothing.
D.report(box, ['x'], []);
assert.strictEqual(box.children.length, 1);
D.report(box, ['x'], ['w1', 'w2']);
const ul = box.children[1];
assert.strictEqual(ul.tagName, 'ul');
assert.strictEqual(ul.className, 'warn-list');
assert.deepStrictEqual(ul.children.map(c => c.textContent), ['w1', 'w2']);
// More than 50 (the default cap): 50 shown and "... n more".
const many = Array.from({length: 53}, (_, i) => 'w' + i);
D.report(box, [], many);
const items = box.children[0].children.map(c => c.textContent);
assert.strictEqual(items.length, 51);
assert.strictEqual(items[49], 'w49');
assert.strictEqual(items[50], '... 3 more');
// Own cap.
D.report(box, [], many, 2);
assert.deepStrictEqual(box.children[0].children.map(c => c.textContent), ['w0', 'w1', '... 51 more']);
console.log('ok');
"#);
}
