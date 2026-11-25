//! The Review tab's job following and Bulk tools (`assets/review-bulk.js`) on a fake page: the
//! line a followed task shows, the job picked up at load, its end (report, reload, the overlap
//! check's "Show the overlapping frames"), and the bulk flow (dry run with the filters, confirm,
//! run, report, reload; nothing to do; no; a refused score; failures; a memory refusal; an error).
//!
//! [`DRIVER`] was first run against the bulk section of `assets/review.js` before it moved to its
//! own file (characterisation): the requests, confirm texts and report lines are the same.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI), which
//! makes a missing `node` a failure: a skip must not read as a pass there.
#![allow(clippy::unwrap_used)]

use crate::js_harness::{asset, have_node, run_node_ok};

/// `process.argv[2]`: the module.
const DRIVER: &str = r#"
const assert = require('assert');
const flush = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); };

// A fake page: elements keep their children and listeners; `report` keeps its lines.
const node = (tag, cls, text) => ({
  tag, className: cls || '', text: text == null ? '' : text, children: [], listeners: {}, hidden: true, value: 0, disabled: false,
  appendChild(c) { this.children.push(c); },
  addEventListener(type, f) { (this.listeners[type] = this.listeners[type] || []).push(f); },
  click() { (this.listeners.click || []).forEach((f) => f({})); }
});
const report = (box, lines, warnings) => { box.lines = lines.slice(); box.warnings = warnings || null; box.children = []; };
const button = (text, fn) => ({ tag: 'button', text, fn });
const ids = {};
const $ = (id) => ids[id] || null;
for (const id of ['rv-al-status', 'rv-al-progress', 'rv-accept-auto', 'rv-accept-auto-msg', 'rv-accept-repeats',
  'rv-reject-low', 'rv-reject-low-msg', 'rv-reject-low-score', 'rv-reject-other', 'rv-reject-other-msg',
  'rv-suggest-overlaps', 'rv-suggest-msg', 'rv-suggest-progress', 'rv-suggest-anyway', 'rv-suggest-large',
  'rv-suggest-again', 'rv-reject-stacked', 'rv-reject-stacked-msg', 'rv-restore-reopened', 'rv-restore-reopened-msg']) ids[id] = node('x');
const msg = node('p');
const filtersForm = { elements: { bucket: { value: 'b1' }, scene: { value: '' }, camera: { value: 'gate' }, label: { value: 'cat,dog' }, source: { value: 'queue' } } };

// Requests: GET through fetch (or PrismPoll.fetch inside a poll), POST through postJson.
const gets = [], posts = [];
let getAnswers = [], postAnswers = [];
const answer = (q) => { const a = q.shift(); return a instanceof Error ? Promise.reject(a) : Promise.resolve(a); };
globalThis.fetch = (url, init) => { gets.push({ url, init, poll: false }); return answer(getAnswers).then((j) => ({ ok: true, status: 200, json: () => Promise.resolve(j) })); };
let inPoll = false, polls = [], background = 0;
globalThis.PrismPoll = {
  inPoll: () => inPoll,
  fetch: (url, init) => { gets.push({ url, init, poll: true }); return answer(getAnswers).then((j) => ({ ok: true, status: 200, json: () => Promise.resolve(j) })); },
  create: (o) => { const p = { o, started: 0, stopped: 0, start() { this.started++; }, stop() { this.stopped++; } }; polls.push(p); return p; },
  background: (fn) => { background++; inPoll = true; try { return fn(); } finally { inPoll = false; } }
};
let ticked = false, synced = [];
globalThis.PrismMemory = { ticked: () => ticked, sync: (row, m) => synced.push([row, m]) };
globalThis.window = globalThis;
let confirms = [], confirmAnswer = true;
globalThis.confirm = (t) => { confirms.push(t); return confirmAnswer; };
const postJson = (url, body) => { posts.push({ url, body: JSON.parse(JSON.stringify(body)) }); return answer(postAnswers); };
let reloads = 0, shown = 0, filtered = 0, forgot = 0;
const ctx = {
  $, el: node, report, postJson, button, msg, filtersForm,
  reloadQueue: () => { reloads++; return Promise.resolve(); },
  showOverlaps: () => { shown++; }, filterReload: () => { filtered++; }, forgetHistories: () => { forgot++; }
};

require(process.argv[2]);
const B = globalThis.ReviewBulk;

(async () => {
  // describeTask: the line of a queued, running or paused task.
  assert.strictEqual(B.describeTask({ state: 'queued' }), 'Queued: auto-label (waiting for the queue).');
  assert.strictEqual(B.describeTask({ state: 'queued', title: 'Overlap check' }), 'Queued: Overlap check (waiting for the queue).');
  assert.strictEqual(B.describeTask({ state: 'running' }), 'starting');
  assert.strictEqual(B.describeTask({ state: 'running', title: 'Auto-label', stage: 'label_frames', step: 3, steps: 10, percent: 30, detail: 'gate' }),
    'Auto-label: label frames (3/10), 30%: gate');
  assert.strictEqual(B.describeTask({ state: 'paused', stage: 'x', percent: 0 }), 'x, 0%');

  // At load: the busy tasks; a running overlap check wins over a queued auto-label.
  getAnswers.push({ tasks: [{ id: 't1', kind: 'autolabel', state: 'queued' }, { id: 't0', kind: 'import', state: 'running' }, { id: 't2', kind: 'overlaps', state: 'running' }] });
  const api = B.init(ctx);
  await flush();
  assert.deepStrictEqual(gets.map((g) => [g.url, g.poll]), [['/v1/train/jobs?tasks=busy', false]]);
  assert.deepStrictEqual(gets[0].init, { headers: { Accept: 'application/json' }, cache: 'no-store' });
  assert.strictEqual(polls.length, 1);
  assert.strictEqual(polls[0].o.period, 3000);
  assert.strictEqual(polls[0].started, 1);
  assert.strictEqual(polls[0].o.more(false), false);
  assert.strictEqual(polls[0].o.more(true), true);
  assert.strictEqual(polls[0].o.more(undefined), true);

  // A tick asked from a poll asks through the poll; running: the line and the bar.
  const box = ids['rv-suggest-msg'], bar = ids['rv-suggest-progress'];
  getAnswers.push({ task: { id: 't2', state: 'running', title: 'Overlap check', stage: 'checking_pairs', step: 3, steps: 10, percent: 30 } });
  inPoll = true;
  let go = polls[0].o.run();
  inPoll = false;
  assert.strictEqual(await go, true);
  assert.deepStrictEqual(gets[1].url, '/v1/train/jobs?task=t2');
  assert.strictEqual(gets[1].poll, true);
  assert.deepStrictEqual(box.lines, ['Overlap check: checking pairs (3/10), 30%']);
  assert.strictEqual(bar.hidden, false);
  assert.strictEqual(bar.value, 30);
  // No such task (yet): keep asking. An answer without the field is an error.
  getAnswers.push({ task: null });
  assert.strictEqual(await polls[0].o.run(), true);
  getAnswers.push({ tasks: [] });
  await assert.rejects(polls[0].o.run(), /the jobs answer has no task field/);
  // Done: the line, green, the bar hidden, the result's button, a reload as a poll; stop.
  getAnswers.push({ task: { id: 't2', state: 'done', summary: '12 suggestions' } });
  assert.strictEqual(await polls[0].o.run(), false);
  assert.strictEqual(polls[0].stopped, 1);
  assert.deepStrictEqual(box.lines, ['Done: 12 suggestions']);
  assert.strictEqual(box.className, 'review-report ok-text');
  assert.strictEqual(bar.hidden, true);
  assert.strictEqual(background, 1);
  assert.strictEqual(reloads, 1);
  assert.deepStrictEqual(box.children.map((c) => c.text), ['Show the overlapping frames']);
  box.children[0].fn();
  assert.deepStrictEqual([shown, filtered], [1, 1]);
  // A tick of a task no longer followed stops.
  getAnswers.push({ task: { id: 't2', state: 'running' } });
  assert.strictEqual(await polls[0].o.run(), false);

  // An auto-label task followed on the line above the frame; failed and cancelled lines.
  api.pollTask('a1');
  assert.strictEqual(polls.length, 2);
  getAnswers.push({ task: { id: 'a1', state: 'failed', error: 'out of memory' } });
  assert.strictEqual(await polls[1].o.run(), false);
  assert.deepStrictEqual(ids['rv-al-status'].lines, ['Failed: out of memory']);
  assert.strictEqual(ids['rv-al-status'].className, 'review-report error-text');
  assert.strictEqual(reloads, 2);
  api.pollTask('a2');
  getAnswers.push({ task: { id: 'a2', state: 'cancelled' } });
  await polls[2].o.run();
  assert.deepStrictEqual(ids['rv-al-status'].lines, ['Cancelled.']);
  api.pollTask('a3');
  assert.strictEqual(polls[2].stopped, 1, 'following another task stops the last');
  getAnswers.push({ task: { id: 'a3', state: 'cancelled', summary: '3 of 9' } });
  await polls[3].o.run();
  assert.deepStrictEqual(ids['rv-al-status'].lines, ['Cancelled: 3 of 9']);

  // Accept auto: the dry run with the filters set, the confirm, the run, the report, a reload.
  const acc = ids['rv-accept-auto'], accBox = ids['rv-accept-auto-msg'];
  postAnswers.push({ eligible: 3 }, { accepted: 3, failed: [] });
  reloads = 0;
  acc.click();
  assert.strictEqual(acc.disabled, true);
  assert.deepStrictEqual(accBox.lines, ['Counting the frames...']);
  await flush();
  assert.deepStrictEqual(posts, [
    { url: '/v1/review/accept-auto', body: { bucket: 'b1', camera: 'gate', label: 'cat,dog', source: 'queue', dry_run: true } },
    { url: '/v1/review/accept-auto', body: { bucket: 'b1', camera: 'gate', label: 'cat,dog', source: 'queue', dry_run: false } }
  ]);
  assert.deepStrictEqual(confirms, ['Accept 3 frame(s) whose boxes the auto-labeler marked accept? They are saved as reviewed: those boxes accepted, unconfirmed hint boxes rejected. Frames with a box flagged for review are left alone.']);
  assert.deepStrictEqual(accBox.lines, ['Accepted 3 frame(s).']);
  assert.deepStrictEqual(accBox.warnings, []);
  assert.strictEqual(accBox.className, 'review-report ok-text');
  assert.strictEqual(reloads, 1);
  assert.strictEqual(acc.disabled, false);
  // Nothing to do: the server's message, no confirm.
  posts.length = 0; confirms.length = 0;
  postAnswers.push({ eligible: 0, message: 'Nothing here.' });
  acc.click();
  await flush();
  assert.strictEqual(posts.length, 1);
  assert.deepStrictEqual(confirms, []);
  assert.deepStrictEqual(accBox.lines, ['Nothing here.']);
  // No to the confirm.
  posts.length = 0;
  confirmAnswer = false;
  postAnswers.push({ eligible: 2 });
  acc.click();
  await flush();
  assert.strictEqual(posts.length, 1);
  assert.deepStrictEqual(accBox.lines, ['Nothing accepted.']);
  confirmAnswer = true;
  // An error: red, the button back.
  postAnswers.push(new Error('HTTP 500'));
  acc.click();
  await flush();
  assert.deepStrictEqual(accBox.lines, ['HTTP 500']);
  assert.strictEqual(accBox.className, 'review-report error-text');
  assert.strictEqual(acc.disabled, false);

  // Accept repeats reports on the message line when it has no box of its own.
  posts.length = 0;
  postAnswers.push({ eligible: 0 });
  ids['rv-accept-repeats'].click();
  await flush();
  assert.deepStrictEqual(msg.lines, ['No repeat-only frame matches the filters (a frame with a box still to decide never is).']);

  // Reject low: a score out of range never asks the server; failures listed, red.
  const low = ids['rv-reject-low'], lowBox = ids['rv-reject-low-msg'];
  posts.length = 0; confirms.length = 0;
  ids['rv-reject-low-score'].value = '0';
  low.click();
  assert.deepStrictEqual(lowBox.lines, ['Enter a score between 0.01 and 1.']);
  assert.strictEqual(lowBox.className, 'review-report error-text');
  assert.strictEqual(posts.length, 0);
  ids['rv-reject-low-score'].value = '0.3';
  postAnswers.push({ boxes: 5, eligible: 2 }, { boxes: 4, failed: ['a.jpg: locked', { frame: 'b.jpg' }] });
  low.click();
  await flush();
  assert.deepStrictEqual(posts[0].body, { min_score: 0.3, bucket: 'b1', camera: 'gate', label: 'cat,dog', source: 'queue', dry_run: true });
  assert.deepStrictEqual(confirms, ['Reject 5 box(es) on 2 frame(s): the pending auto boxes scoring below 0.3? They are kept as hard negatives (reason low_score); the frames stay unreviewed and a reviewer can still change them per frame.']);
  assert.deepStrictEqual(lowBox.lines, ['Rejected 4 box(es).']);
  assert.deepStrictEqual(lowBox.warnings, ['a.jpg: locked', '{"frame":"b.jpg"}']);
  assert.strictEqual(lowBox.className, 'review-report error-text');

  // Unwanted classes: renamed boxes in the confirm, then the reviewed frames asked about.
  const other = ids['rv-reject-other'], otherBox = ids['rv-reject-other-msg'];
  posts.length = 0; confirms.length = 0;
  postAnswers.push({ boxes: 6, relabelled: 2, eligible: 4 },
    { boxes: 6, message: 'Changed 6.', reviewed_with_unwanted: { frames: 2, boxes: 3, classes: ['kite', 'boat'] } },
    { boxes: 3, message: 'Rejected 3.', failed: ['c.jpg'] });
  other.click();
  await flush();
  assert.deepStrictEqual(posts.map((p) => p.url), ['/v1/review/reject-other-class', '/v1/review/reject-other-class', '/v1/review/reject-unwanted-reviewed']);
  assert.deepStrictEqual(posts[2].body, {});
  assert.deepStrictEqual(confirms, [
    'Reject 4 pending model / auto box(es) of unwanted classes (hidden, or not in the class list) on 4 frame(s)? They are kept as evidence (reason other_class); the frames stay unreviewed and a reviewer can still change them per frame. 2 box(es) whose label is a class of the list under another spelling (e.g. pottedplant) are renamed to the list\'s spelling instead.',
    '2 reviewed frame(s) have 3 accepted box(es) of kite, boat. Reject those too? They are kept as evidence (reason other_class) and the frames stay reviewed; Cancel keeps them as training labels.'
  ]);
  assert.deepStrictEqual(otherBox.lines, ['Changed 6.']);
  assert.deepStrictEqual(otherBox.children.map((c) => [c.className, c.text]), [['', 'Rejected 3.'], ['error-text', 'c.jpg']]);
  // Nothing pending, reviewed frames left as they are.
  confirmAnswer = false;
  postAnswers.push({ boxes: 0, reviewed_with_unwanted: { frames: 1, boxes: 1, classes: ['kite'] } });
  other.click();
  await flush();
  assert.deepStrictEqual(otherBox.lines, ['No pending box of an unwanted class (hidden, or not in the class list) on the unreviewed frames the filters select.']);
  assert.deepStrictEqual(otherBox.children.map((c) => c.text), ['Reviewed frames left as they are (1 with boxes of kite).']);
  confirmAnswer = true;

  // Suggest overlaps: no `source`; a memory refusal stops unless Start anyway is ticked; queued:
  // followed.
  const sug = ids['rv-suggest-overlaps'];
  posts.length = 0; confirms.length = 0;
  ids['rv-suggest-large'].checked = true;
  postAnswers.push({ pairs: 4, frames: 2, crops: 8, memory: { refusal: 'Too little memory.' } });
  sug.click();
  await flush();
  assert.deepStrictEqual(posts[0].body, { bucket: 'b1', camera: 'gate', label: 'cat,dog', allow_large: true, dry_run: true });
  assert.deepStrictEqual(synced, [[ids['rv-suggest-anyway'], { refusal: 'Too little memory.' }]]);
  assert.deepStrictEqual(box.lines, ['Too little memory.', 'To queue it anyway, tick Start anyway above, then press Suggest overlaps again.']);
  assert.deepStrictEqual(confirms, []);
  ticked = true;
  ids['rv-suggest-again'].checked = true;
  posts.length = 0;
  const before = polls.length;
  postAnswers.push({ pairs: 4, frames: 2, crops: 800, memory: { refusal: 'Too little memory.' } }, { message: 'Queued.', id: 'o9' });
  sug.click();
  await flush();
  assert.deepStrictEqual(posts[1].body, { bucket: 'b1', camera: 'gate', label: 'cat,dog', allow_large: true, again: true, dry_run: false, ignore_memory_check: true });
  assert.deepStrictEqual(confirms, ['Warning: Too little memory.\n\nCheck 4 overlapping pair(s) on 2 frame(s), reviewed frames included, answered pairs asked again, with RF-DETR L (800 crops, on the training queue, about 2 min)? It writes suggestions only: no box changes until you confirm one here.']);
  assert.deepStrictEqual(box.lines, ['Queued.']);
  assert.strictEqual(polls.length, before + 1, 'the queued check is followed');

  // Stacked duplicates: no `source`; the camera histories are forgotten after.
  posts.length = 0;
  postAnswers.push({ boxes: 2, eligible: 1 }, { boxes: 2 });
  ids['rv-reject-stacked'].click();
  await flush();
  assert.deepStrictEqual(posts[0].body, { bucket: 'b1', camera: 'gate', label: 'cat,dog', dry_run: true });
  assert.deepStrictEqual(ids['rv-reject-stacked-msg'].lines, ['Rejected 2 box(es).']);
  assert.strictEqual(forgot, 1);

  // Restore reopened.
  posts.length = 0;
  postAnswers.push({ eligible: 1 }, { restored: 1 });
  ids['rv-restore-reopened'].click();
  await flush();
  assert.strictEqual(posts[0].url, '/v1/review/restore-reopened');
  assert.deepStrictEqual(ids['rv-restore-reopened-msg'].lines, ['Restored 1 frame(s).']);

  // A page without the job line asks for no task at load.
  const n = gets.length;
  B.init(Object.assign({}, ctx, { $: () => null }));
  await flush();
  assert.strictEqual(gets.length, n);
  console.log('ok');
})().catch((e) => { console.error(e); process.exit(1); });
"#;

#[test]
fn bulk_tools_dry_run_confirm_run_and_follow_their_jobs() {
    if !have_node() {
        return;
    }
    run_node_ok(DRIVER, &[asset("review-bulk.js").as_os_str()]);
}
