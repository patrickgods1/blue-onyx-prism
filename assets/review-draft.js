// Review tab, the choices about drafts (/train/review; loaded after live.js, before review.js).
// A draft is the frame's unsaved edits that review.js keeps in sessionStorage (PrismDraft,
// live.js) when the page goes out of view, so a tab the phone discarded comes back with them.
// These are the decisions only: what a hidden page keeps, what the page's first frame does with a
// draft, which frame the page opens with. Pure (no page, no storage: review.js reads and writes
// PrismDraft); tests/it/js_review_draft.rs runs this file in Node.
(function () {
  'use strict';

  // When the page goes out of view: 'skip' while the draft's own frame is loading (nothing is
  // written or dropped meanwhile), 'save' a frame with unsaved edits, else 'drop'.
  function keepChoice(waiting, cur, dirty) {
    if (waiting) return 'skip';
    return cur && dirty ? 'save' : 'drop';
  }

  // The page's first frame `id` at revision `rev` is in, and `draft` is what storage holds:
  //   'none'     no draft: nothing to drop or say
  //   'other'    another frame's: dropped, nothing said
  //   'stale'    this frame, saved by someone since (or the draft has no box list): dropped, and
  //              the page says the edits were dropped (nobody's edits are overwritten unseen)
  //   'restore'  this frame at the same revision: dropped from storage, its labels back on screen
  function restoreChoice(draft, id, rev) {
    var v = globalThis.PrismDraft.verdict(draft, id, rev);
    if (v === 'none' || v === 'other') return v;
    if (v === 'drop' || !draft.labels || !Array.isArray(draft.labels.boxes)) return 'stale';
    return 'restore';
  }

  // The box the draft had selected, while it is one of the `n` boxes restored; else none (-1).
  function selectedOf(draft, n) {
    return (typeof draft.selected === 'number' && draft.selected >= 0 && draft.selected < n) ? draft.selected : -1;
  }

  // The frame a link names (`?frame=<scene>/<camera>/<date>/<stem>.jpg`, the Refine tab's "Check
  // these"): null for none, one over 400 characters or one that is no .jpg.
  function linkFrame(v) {
    return !v || v.length > 400 || !/\.jpg$/.test(v) ? null : v;
  }

  // Where the page starts: a draft's frame ('draft'), else the link's ('link'), else the queue's
  // first ('queue', no frame).
  function start(draft, link) {
    if (draft && typeof draft.id === 'string' && draft.id) return { from: 'draft', frame: draft.id };
    if (link) return { from: 'link', frame: link };
    return { from: 'queue', frame: null };
  }

  globalThis.ReviewDraft = {
    keepChoice: keepChoice, restoreChoice: restoreChoice, selectedOf: selectedOf, linkFrame: linkFrame, start: start
  };
})();
