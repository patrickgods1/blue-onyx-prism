//! The Review tab's decisions about drafts (`assets/review-draft.js`, with `PrismDraft.verdict`
//! of `assets/live.js`): what a hidden page keeps, what the first frame does with a draft (none,
//! another frame's, stale, restore; the selection), the frame a link names and where the page
//! starts.
//!
//! Before the move, review.js's `keepDraft`, `restoreDraft` and its start were run against the new
//! ones on every case below with the page parts faked (same storage calls, same messages).
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI), which
//! makes a missing `node` a failure: a skip must not read as a pass there.
#![allow(clippy::unwrap_used)]

use crate::js_harness::{asset, have_node, run_node_ok};

/// `process.argv[2]`: live.js; `[3]`: the module.
const DRIVER: &str = r#"
const assert = require('assert');
require(process.argv[2]);
require(process.argv[3]);
const D = globalThis.ReviewDraft;

// keepChoice: nothing while the draft's own frame loads; unsaved edits kept; else dropped.
assert.strictEqual(D.keepChoice(true, { frame: 'a' }, true), 'skip');
assert.strictEqual(D.keepChoice(true, null, false), 'skip');
assert.strictEqual(D.keepChoice(false, { frame: 'a' }, true), 'save');
assert.strictEqual(D.keepChoice(false, { frame: 'a' }, false), 'drop');
assert.strictEqual(D.keepChoice(false, null, true), 'drop');

// restoreChoice: by frame and revision; a draft without a box list is never put on screen.
const labels = { boxes: [{ label: 'cat' }, { label: 'dog' }] };
assert.strictEqual(D.restoreChoice(null, 'a.jpg', '1'), 'none');
assert.strictEqual(D.restoreChoice('text', 'a.jpg', '1'), 'none');
assert.strictEqual(D.restoreChoice({ id: 'b.jpg', rev: '1', labels }, 'a.jpg', '1'), 'other');
assert.strictEqual(D.restoreChoice({ id: 'a.jpg', rev: '1', labels }, 'a.jpg', '2'), 'stale', 'saved by someone since');
assert.strictEqual(D.restoreChoice({ id: 'a.jpg', rev: '1', labels }, 'a.jpg', '1'), 'restore');
assert.strictEqual(D.restoreChoice({ id: 'a.jpg', rev: '', labels }, 'a.jpg', ''), 'restore', 'never saved, still never saved');
assert.strictEqual(D.restoreChoice({ id: 'a.jpg', rev: '1' }, 'a.jpg', '1'), 'stale', 'no labels');
assert.strictEqual(D.restoreChoice({ id: 'a.jpg', rev: '1', labels: {} }, 'a.jpg', '1'), 'stale', 'no box list');
assert.strictEqual(D.restoreChoice({ id: 'a.jpg', rev: '1', labels: { boxes: 'x' } }, 'a.jpg', '1'), 'stale', 'boxes not a list');
assert.strictEqual(D.restoreChoice({ id: 'a.jpg', rev: '1', labels: { boxes: [] } }, 'a.jpg', '1'), 'restore', 'an empty frame is a frame');

// selectedOf: the draft's selection while it is one of the boxes.
assert.strictEqual(D.selectedOf({ selected: 1 }, 2), 1);
assert.strictEqual(D.selectedOf({ selected: 0 }, 2), 0);
assert.strictEqual(D.selectedOf({ selected: 2 }, 2), -1);
assert.strictEqual(D.selectedOf({ selected: -1 }, 2), -1);
assert.strictEqual(D.selectedOf({ selected: '1' }, 2), -1);
assert.strictEqual(D.selectedOf({}, 2), -1);
assert.strictEqual(D.selectedOf({ selected: 0 }, 0), -1);

// linkFrame: a .jpg of at most 400 characters.
assert.strictEqual(D.linkFrame(null), null);
assert.strictEqual(D.linkFrame(''), null);
assert.strictEqual(D.linkFrame('day/gate/2026-11-01/a.jpg'), 'day/gate/2026-11-01/a.jpg');
assert.strictEqual(D.linkFrame('day/gate/2026-11-01/a.png'), null);
assert.strictEqual(D.linkFrame('day/gate/a.jpg?x'), null);
assert.strictEqual(D.linkFrame('x'.repeat(396) + '.jpg'), 'x'.repeat(396) + '.jpg');
assert.strictEqual(D.linkFrame('x'.repeat(397) + '.jpg'), null);

// start: a draft's frame first, then the link's, else the queue.
assert.deepStrictEqual(D.start({ id: 'a.jpg' }, 'b.jpg'), { from: 'draft', frame: 'a.jpg' });
assert.deepStrictEqual(D.start({ id: '' }, 'b.jpg'), { from: 'link', frame: 'b.jpg' });
assert.deepStrictEqual(D.start({ id: 7 }, null), { from: 'queue', frame: null });
assert.deepStrictEqual(D.start(null, 'b.jpg'), { from: 'link', frame: 'b.jpg' });
assert.deepStrictEqual(D.start(null, null), { from: 'queue', frame: null });
console.log('ok');
"#;

#[test]
fn a_draft_comes_back_only_on_its_frame_at_its_revision() {
    if !have_node() {
        return;
    }
    run_node_ok(
        DRIVER,
        &[
            asset("live.js").as_os_str(),
            asset("review-draft.js").as_os_str(),
        ],
    );
}
