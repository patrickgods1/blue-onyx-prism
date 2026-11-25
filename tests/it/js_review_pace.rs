//! The Review tab's session pace (`assets/review-pace.js`): the median, the share accepted, boxes a
//! minute, the "Going fast" rule and the session line with its HUD and break note, on exact
//! numbers and a fake page and clock.
//!
//! [`PAGE`] was first run against the pace section of `assets/review.js` before it moved to its
//! own file (characterisation): the session line, the HUD slot and the chips are the same words.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI), which
//! makes a missing `node` a failure: a skip must not read as a pass there.
#![allow(clippy::unwrap_used)]

use crate::js_harness::{asset, have_node, run_node_ok};

/// Loads the module (`process.argv[2]`) with a clock the driver sets and no real timer.
const HEAD: &str = r#"
const assert = require('assert');
let now = 1000000;
Date.now = () => now;
let intervals = [];
globalThis.setInterval = (f, ms) => { intervals.push(ms); return intervals.length; };
require(process.argv[2]);
const P = globalThis.ReviewPace;
"#;

/// The maths and the page part as review.js had them.
const PAGE: &str = r#"
// median: the middle value (the mean of the middle two), 0 for none; the input stays as it was.
assert.strictEqual(P.median([]), 0);
assert.strictEqual(P.median([3]), 3);
assert.strictEqual(P.median([5, 1, 3]), 3);
const four = [4, 1, 3, 2];
assert.strictEqual(P.median(four), 2.5);
assert.deepStrictEqual(four, [4, 1, 3, 2], 'not sorted in place');
assert.strictEqual(P.median([10, 9, 100]), 10, 'numbers, not text order');

// acceptRate: accepted over decided, null with no decision.
assert.strictEqual(P.acceptRate([]), null);
assert.strictEqual(P.acceptRate([{ acc: 0, rej: 0 }]), null);
assert.strictEqual(P.acceptRate([{ acc: 3, rej: 1 }, { acc: 1, rej: 3 }]), 0.5);
assert.strictEqual(P.acceptRate([{ acc: 2, rej: 0 }]), 1);
assert.deepStrictEqual(P.seconds([{ s: 4 }, { s: 1.5 }]), [4, 1.5]);

// boxesPerMinute: '' under a second of review, one decimal under 10, whole numbers from 10.
assert.strictEqual(P.boxesPerMinute([]), '');
assert.strictEqual(P.boxesPerMinute([{ s: 0.9, boxes: 5 }]), '');
assert.strictEqual(P.boxesPerMinute([{ s: 30, boxes: 2 }]), '4.0');
assert.strictEqual(P.boxesPerMinute([{ s: 20, boxes: 1 }, { s: 40, boxes: 2 }]), '3.0');
assert.strictEqual(P.boxesPerMinute([{ s: 60, boxes: 25 }]), '25');
assert.strictEqual(P.boxesPerMinute([{ s: 60, boxes: 10 }]), '10');
assert.strictEqual(P.boxesPerMinute([{ s: 500, boxes: 83 }]), '10.0', '9.96 a minute rounds up under 10');
assert.strictEqual(P.boxesPerMinute([{ s: 60 }]), '0.0', 'a frame without boxes counts none');

// The page part on a fake page.
const made = [];
const el = (tag, cls, text) => {
  const e = { tag, cls, text: text || '', children: [], appendChild(c) { this.children.push(c); } };
  made.push(e);
  return e;
};
const button = (text, fn, cls) => ({ button: text, fn, cls });
// Setting the text drops the children, as on a page.
const line = {
  text: 'old', children: [], appendChild(c) { this.children.push(c); },
  get textContent() { return this.text; }, set textContent(t) { this.text = t; this.children = []; }
};
const details = { textContent: 'old' };
const hud = [];
const hudPace = { name: 'pace' };
const hudSet = (e, t) => { assert.strictEqual(e, hudPace); hud.push(t); };
const pace = P.init({ el, button, sessionLine: line, detailsPace: details, hudSet, hudPace });
assert.deepStrictEqual(intervals, [30000], 'the line is drawn again every 30 s');
const words = () => line.children.map((c) => c.text);

// No frame yet: nothing.
pace.show();
assert.strictEqual(line.textContent, '');
assert.deepStrictEqual(line.children, []);
assert.strictEqual(details.textContent, '');
assert.deepStrictEqual(hud, ['']);

// One frame, 30 s from load to save, two accepted, one rejected, one drawn.
now += 30000;
pace.noteSaved({ accepted: 2, rejected: 1, decided: 3, drawn: 1 }, now - 30000);
assert.deepStrictEqual(words(), ['Session 1 frame · 4 boxes, 8.0 a minute · 67% accepted · 0 min']);
assert.strictEqual(line.children[0].cls, 'rv-session-stats');
assert.strictEqual(details.textContent, 'Session 1 frame · 4 boxes, 8.0 a minute · 67% accepted · 0 min');
assert.strictEqual(hud[hud.length - 1], '4 boxes · 8.0/min');

// A frame left open 20 minutes counts 10; one box: "box", no decision: no share accepted.
now += 20 * 60000;
pace.noteSaved({ accepted: 0, rejected: 0, decided: 0, drawn: 0 }, now - 20 * 60000);
assert.deepStrictEqual(words(), ['Session 2 frames · 4 boxes, 0.4 a minute · 67% accepted · 20 min']);
assert.strictEqual(hud[hud.length - 1], '4 boxes · 0.4/min');

// 45 minutes in: the break note, with Later (another 45 minutes).
now += 26 * 60000;
pace.show();
assert.deepStrictEqual(words(), ['Session 2 frames · 4 boxes, 0.4 a minute · 67% accepted · 46 min', 'Take a break? ']);
const brk = line.children[1];
assert.strictEqual(brk.cls, 'rv-chip warn');
assert.strictEqual(brk.children[0].button, 'Later');
assert.strictEqual(brk.children[0].cls, 'secondary');
brk.children[0].fn();
assert.deepStrictEqual(words(), ['Session 2 frames · 4 boxes, 0.4 a minute · 67% accepted · 46 min']);
now += 45 * 60000;
pace.show();
assert.strictEqual(words().length, 2, 'the note again 45 minutes after Later');

// Going fast: 10 slow frames (20 s, half accepted), then 10 fast ones (2 s, all accepted).
const fast = P.init({ el, button, sessionLine: line, detailsPace: null, hudSet, hudPace });
let start = now;
for (let i = 0; i < 10; i++) { now += 20000; fast.noteSaved({ accepted: 1, rejected: 1, decided: 2, drawn: 0 }, now - 20000); }
for (let i = 0; i < 4; i++) { now += 2000; fast.noteSaved({ accepted: 1, rejected: 0, decided: 1, drawn: 0 }, now - 2000); }
assert.strictEqual(words().length, 1, '14 frames: too few to say');
for (let i = 0; i < 6; i++) { now += 2000; fast.noteSaved({ accepted: 1, rejected: 0, decided: 1, drawn: 0 }, now - 2000); }
assert.deepStrictEqual(words(), ['Session 20 frames · 30 boxes, 8.2 a minute · 67% accepted · 3 min', 'Going fast: check the small boxes']);
assert.strictEqual(line.children[1].cls, 'rv-chip warn');
assert.ok(now - start < 45 * 60000);

// No line on the page: nothing drawn, nothing set.
const before = hud.length;
P.init({ el, button, sessionLine: null, detailsPace: details, hudSet, hudPace }).show();
assert.strictEqual(hud.length, before);
"#;

/// The rules pulled out of the page part when it moved.
const PURE: &str = r#"
// frameRecord: load to save, capped at 10 minutes; drawn boxes count as decisions.
assert.deepStrictEqual(P.frameRecord({ accepted: 2, rejected: 1, decided: 3, drawn: 1 }, 1500), { s: 1.5, acc: 2, rej: 1, boxes: 4 });
assert.strictEqual(P.frameRecord({ accepted: 0, rejected: 0, decided: 0, drawn: 0 }, 700000).s, 600);

// goingFast: from 15 frames, the last 10 under 0.4 of the session's median time and accepting more.
const f = (s, acc, rej) => ({ s, acc, rej, boxes: acc + rej });
const slow = Array.from({ length: 10 }, () => f(20, 1, 1));
const quick = Array.from({ length: 10 }, () => f(2, 1, 0));
assert.strictEqual(P.goingFast(slow.concat(quick)), true);
assert.strictEqual(P.goingFast(slow.slice(0, 4).concat(quick)), false, '14 frames');
assert.strictEqual(P.goingFast(slow.slice(0, 5).concat(quick)), false, 'median of the session is 2 s: not faster');
assert.strictEqual(P.goingFast(slow.concat(Array.from({ length: 10 }, () => f(2, 1, 1)))), false, 'fast, accepting the same share');
assert.strictEqual(P.goingFast(slow.concat(Array.from({ length: 10 }, () => f(2, 0, 0)))), false, 'fast, deciding nothing');
assert.strictEqual(P.goingFast(slow.concat(Array.from({ length: 10 }, () => f(8.1, 1, 0)))), false, '8.1 s is not under 0.4 x 14.05');
assert.strictEqual(P.goingFast(slow.concat(Array.from({ length: 10 }, () => f(6, 1, 0)))), false, '6 s is not under 0.4 x 13');
assert.strictEqual(P.goingFast(slow.concat(Array.from({ length: 10 }, () => f(4.9, 1, 0)))), true, '4.9 s is under 0.4 x 12.45');
// 15 frames is the least: the same rule holds on 14 and does not count.
const mixed = [1, 1, 20, 1, 20, 1, 20, 1, 20, 1].map((s) => f(s, 1, 0));
assert.strictEqual(P.goingFast(slow.slice(0, 5).concat(mixed)), true, '15 frames');
assert.strictEqual(P.goingFast(slow.slice(0, 4).concat(mixed)), false, '14 frames');

// sessionText: singular words, no pace without boxes, no share without decisions.
assert.deepStrictEqual(P.sessionText([f(30, 1, 0)], 3), { hud: '1 box · 2.0/min', line: 'Session 1 frame · 1 box, 2.0 a minute · 100% accepted · 3 min' });
assert.deepStrictEqual(P.sessionText([{ s: 30, acc: 0, rej: 0, boxes: 0 }], 0), { hud: '0 boxes', line: 'Session 1 frame · 0 boxes · 0 min' });
assert.deepStrictEqual(P.sessionText([{ s: 0.5, acc: 1, rej: 0, boxes: 1 }], 0), { hud: '1 box', line: 'Session 1 frame · 1 box · 100% accepted · 0 min' });
"#;

#[test]
fn the_session_line_counts_boxes_a_minute_and_warns_when_going_fast() {
    if !have_node() {
        return;
    }
    let driver = format!("{HEAD}{PAGE}{PURE}\nconsole.log('ok');\n");
    run_node_ok(&driver, &[asset("review-pace.js").as_os_str()]);
}
