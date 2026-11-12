// Review tab (/train/review): one frame at a time from GET /v1/review/next, its boxes from
// GET /v1/review/labels, drawn on a canvas; decisions saved with POST /v1/review/labels
// (with the loaded `revision`; a 409 means someone else saved the frame meanwhile).
// Filters (bucket, scene, camera, class, auto, status, source) go to /v1/review/next as query
// params; they start from the page's own URL (a deep link such as ?auto=overlaps), stay in it, and
// fold to the one-line summary above the frame, whose "Auto-label these frames" link passes them
// on. The canvas edits boxes (draw, select, move, resize, nudge) under a zoom / pan view.
// A job queued on the Auto-label tab is followed on GET /v1/train/jobs?task=<id> (its record only;
// GET /v1/training on a server without it), every 3 s while the tab shows, and the frames reload
// when it ends. Under Bulk tools, "Accept auto: accept frames" calls POST /v1/review/accept-auto and
// "Reject auto boxes below" POST /v1/review/reject-low, "Reject boxes of unwanted classes"
// POST /v1/review/reject-other-class, "Restore reopened frames" POST /v1/review/restore-reopened
// and "Reject stacked duplicates" POST /v1/review/reject-stacked
// (all five: dry run, confirm, run). *Suggest overlaps...* queues the overlap check
// (POST /v1/review/suggest-overlaps, followed like an auto-label job); its stored suggestions show on
// the frame, and `y` accepts the one on screen. The dataset check's suggestions (`labels.refine`,
// `refine_live`; Auto filter "dataset check suggestions", `auto=refine[&refine_kind=]`) come after
// them, one at a time, strongest first: drawn on the frame (a dashed proposed box, or the box they
// are about marked and selected), `y` accepts and `w` says no (R.answerRefine), saved with the
// frame by the ordinary save. A labelers run sorts its new boxes into tiers
// (training/autolabel/triage.rs): dismissed ones (state rejected, reason auto_dismissed or, at a
// spot the reviewer rejected, spot_dismissed) are left off the frame and the list until `i` shows
// them, likely ones come first and `c` accepts them, the other flagged ones are drawn quieter.
// `t` accepts a real object that stays put and remembers its spot (POST /v1/review/spots/parked);
// later runs accept it there. GET /v1/review/spots feeds the card's line, and Panel options has
// "Camera moved: forget spots" (POST /v1/review/spots/moved).
// The folder form posts JSON to /v1/review/folder; a new folder reloads the page. Export, X-AnyLabeling
// import and the class list are on the Dataset tab (train-dataset.js).
// Classes: GET /v1/review/class-catalog feeds the class pickers (class-picker.js): the Class
// filter (label=a,b) and "Show boxes of" (display only). Unwanted classes
// (hidden, or not in the list) are what the reviewer does not review: pickers, the new-box class
// select and the 1-0 keys leave the hidden ones out, the canvas / box list / stepping skip boxes
// of hidden classes and rejected boxes of unlisted ones (they stay in the file; a save rejects
// pending proposals of hidden classes), and the server rejects their pending boxes when a class
// becomes unwanted. When the reject button leaves reviewed frames with accepted boxes of unwanted
// classes, the page asks, then POST /v1/review/reject-unwanted-reviewed. Class ids never change.
// No inline handlers (CSP): everything is wired here; server values come from data-*.
(function () {
  'use strict';
  var root = document.getElementById('review');
  if (!root) return;
  var classes = [];
  try { classes = JSON.parse(root.getAttribute('data-classes') || '[]'); } catch (e) { classes = []; }
  var CP = window.ClassPicker || null;
  var CU = CP ? CP.util : null;
  var hiddenClasses = [];  // collect.hidden_classes: not offered by the pickers and shortcuts
  var catalog = null;      // GET /v1/review/class-catalog
  var shownLabels = [];    // "Show boxes of": empty = every box
  // The pure rules (review-rules.js, tested against their Rust twins by tests/it/js_rules.rs).
  var R = window.ReviewRules;
  var resolved = R.resolved, isDismissal = R.isDismissal, endDismissal = R.endDismissal;
  var iou = R.iou, sameObject = R.sameObject;
  var markRejected = R.markRejected, markDuplicate = R.markDuplicate;

  var $ = function (id) { return document.getElementById(id); };
  var canvas = $('rv-canvas');
  var ctx = canvas.getContext('2d');
  var filtersForm = $('rv-filters');
  var classSel = $('rv-class');
  var boxList = $('rv-boxes');
  var emptyBox = $('rv-empty');
  var msg = $('rv-msg');
  var fitBtn = $('rv-fit');
  var zoomText = $('rv-zoom');
  var cropWrap = $('rv-crop-wrap');
  var phoneHead = $('rv-phone-head');
  var phoneLine = $('rv-phone-line');
  var likelyBar = $('rv-accept-likely');
  var cropCanvas = $('rv-crop');
  var cropCap = $('rv-crop-cap');
  var grid = canvas.closest('.review-grid');

  // Per-viewer preferences (browser storage may be off: then the defaults).
  function pref(key, dflt) {
    try { var v = window.localStorage.getItem('bop.review.' + key); return v == null ? dflt : v === '1'; } catch (e) { return dflt; }
  }
  function setPref(key, on) {
    try { window.localStorage.setItem('bop.review.' + key, on ? '1' : '0'); } catch (e) { /* not kept */ }
  }

  // The decision bar (the selected box's question) sits above the box list; an older page
  // without it gets one made here.
  var decide = $('rv-decide');
  if (!decide) {
    decide = document.createElement('div');
    decide.id = 'rv-decide';
    boxList.parentNode.insertBefore(decide, boxList);
  }
  decide.classList.add('rv-decide');
  // "Accept likely" and "Show N hidden", beside the box list.
  var boxTools = $('rv-box-tools');
  if (!boxTools) {
    boxTools = document.createElement('div');
    boxTools.id = 'rv-box-tools';
    boxTools.className = 'rv-box-tools';
    boxList.parentNode.insertBefore(boxTools, boxList);
  }
  var zoomBoxBtn = $('rv-zoombox');
  if (!zoomBoxBtn && fitBtn) {
    zoomBoxBtn = document.createElement('button');
    zoomBoxBtn.type = 'button';
    zoomBoxBtn.id = 'rv-zoombox';
    zoomBoxBtn.className = 'small secondary';
    zoomBoxBtn.textContent = 'Zoom to box';
    fitBtn.parentNode.insertBefore(zoomBoxBtn, fitBtn);
  }
  // "Show boxes of", "Class for new boxes" and the key table sit in the collapsed "Panel options"
  // at the panel's foot (the box list takes the height); an older page gets it made here.
  var side = boxList.closest('.review-side');
  if (side && !$('rv-panel-opts')) {
    var opts = document.createElement('details');
    opts.id = 'rv-panel-opts';
    opts.className = 'rv-panel-opts';
    var sum = document.createElement('summary');
    sum.textContent = 'Panel options';
    opts.appendChild(sum);
    [side.querySelector('.rv-show-of'), classSel && classSel.closest('label'), side.querySelector(':scope > details')]
      .forEach(function (e) { if (e) opts.appendChild(e); });
    side.appendChild(opts);
  }
  // The close-up goes first in the side panel.
  if (cropWrap && cropWrap.parentNode && cropWrap !== cropWrap.parentNode.firstElementChild) {
    cropWrap.parentNode.insertBefore(cropWrap, cropWrap.parentNode.firstElementChild);
  }
  // The view-only image setting (`b`), next to Zoom / Fit.
  var imgBtn = $('rv-imgmode');
  if (!imgBtn && fitBtn) {
    imgBtn = el('button', 'small secondary', 'Image');
    imgBtn.type = 'button';
    imgBtn.id = 'rv-imgmode';
    fitBtn.parentNode.insertBefore(imgBtn, zoomBoxBtn || fitBtn);
  }
  // Back (`p`) and the keys overlay (`?`) in the Save row.
  var actionsRow = $('rv-save') && $('rv-save').parentNode;
  var backBtn = $('rv-back');
  if (!backBtn && actionsRow) {
    backBtn = el('button', 'small secondary', 'Back');
    backBtn.type = 'button';
    backBtn.id = 'rv-back';
    actionsRow.insertBefore(backBtn, $('rv-next') || null);
  }
  var keysBtn = $('rv-keys-open');
  if (!keysBtn && actionsRow) {
    keysBtn = el('button', 'small secondary', 'Keys ?');
    keysBtn.type = 'button';
    keysBtn.id = 'rv-keys-open';
    actionsRow.appendChild(keysBtn);
  }
  // The session line (pace, acceptance, time; a break note) under the message line.
  var sessionLine = $('rv-session');
  if (!sessionLine && msg && msg.parentNode) {
    sessionLine = el('p', 'rv-session muted');
    sessionLine.id = 'rv-session';
    msg.parentNode.insertBefore(sessionLine, msg.nextSibling);
  }
  // "Ask about missed objects" in Panel options.
  var missingBox = $('rv-ask-missing');
  var panelOpts = $('rv-panel-opts');
  if (!missingBox && panelOpts) {
    var ml = el('label', 'check');
    missingBox = document.createElement('input');
    missingBox.type = 'checkbox';
    missingBox.id = 'rv-ask-missing';
    ml.appendChild(missingBox);
    ml.appendChild(document.createTextNode(' Ask about missed objects before saving'));
    panelOpts.insertBefore(ml, panelOpts.querySelector('details'));
  }
  // "Camera moved: forget spots" in Panel options.
  if (panelOpts && !$('rv-camera-moved')) {
    var movedBtn = document.createElement('button');
    movedBtn.type = 'button';
    movedBtn.id = 'rv-camera-moved';
    movedBtn.className = 'small secondary';
    movedBtn.textContent = 'Camera moved: forget spots';
    movedBtn.title = "For this frame's camera: parked spots retire and earlier rejections stop hiding boxes. Asks first.";
    movedBtn.addEventListener('click', function () { cameraMoved(); });
    panelOpts.insertBefore(movedBtn, panelOpts.querySelector('details'));
  }
  // The keys overlay.
  var keysDialog = $('rv-keys-dialog');
  if (!keysDialog) {
    keysDialog = document.createElement('dialog');
    keysDialog.id = 'rv-keys-dialog';
    keysDialog.className = 'rv-keys-dialog';
    document.body.appendChild(keysDialog);
  }
  // One column: the overview over the frame while it is zoomed (the whole frame, the boxes, the
  // part the frame below shows); an older page gets one made here.
  var overview = $('rv-overview');
  if (!overview) {
    overview = document.createElement('canvas');
    overview.id = 'rv-overview';
    overview.width = 1;
    overview.height = 1;
    overview.hidden = true;
    overview.setAttribute('aria-label', 'Overview of the whole frame: the outlined part is the one shown below. Tap a box to select it, or tap elsewhere to move the view there.');
    canvas.parentNode.insertBefore(overview, canvas);
  }
  overview.classList.add('rv-overview');
  var stage = canvas.closest('.review-stage');
  // One column: the selected box's main answers in the bar at the screen's foot while boxes still
  // need a decision (renderDecision); empty and hidden otherwise. A child of the grid, not of the
  // panel: stuck to the foot of the screen it may not rise above its container's top, and the
  // panel starts low on a short screen.
  var decideBar = $('rv-decide-bar');
  if (!decideBar && grid) {
    decideBar = el('div', 'rv-decide-bar');
    decideBar.id = 'rv-decide-bar';
    decideBar.hidden = true;
    grid.appendChild(decideBar);
  }
  // The bar's head (pinned with it): the selected box, the frame's progress as a line along the
  // bar's top edge and a count; the answers under it.
  var barHead = decideBar && decideBar.querySelector('.rv-bar-head');
  var barBtns = decideBar && decideBar.querySelector('.rv-bar-btns');
  if (decideBar && !barHead) {
    barHead = el('div', 'rv-bar-head');
    decideBar.appendChild(barHead);
  }
  if (decideBar && !barBtns) {
    barBtns = el('div', 'rv-bar-btns');
    decideBar.appendChild(barBtns);
  }
  // One column: "Boxes (N) and options" folds the box list, the session line and Panel options.
  var listToggle = $('rv-list-toggle');
  if (!listToggle && emptyBox) {
    var emptyLabel = emptyBox.closest('label') || emptyBox;
    listToggle = el('button', 'small secondary rv-list-toggle', 'Boxes and options');
    listToggle.type = 'button';
    listToggle.id = 'rv-list-toggle';
    listToggle.setAttribute('aria-controls', 'rv-boxes');
    emptyLabel.parentNode.insertBefore(listToggle, emptyLabel.nextSibling);
  }
  var listOpen = pref('list', false);

  var cur = null;      // answer of GET /v1/review/labels (with `revision`)
  var img = null;       // the frame image
  var selected = -1;    // index into cur.labels.boxes
  var undo = [];        // snapshots of cur.labels
  var dirty = false;
  var gesture = null;   // the pointer gesture in progress (see the canvas input section)
  var fingers = {};     // pointerId -> canvas point of each finger down on the frame (touch only)
  var drawMode = false; // the Draw toggle (touch UI): a finger's drag draws, taps set corners
  var taps = null;      // two-tap box in progress {kind: 'draw' | 'redraw', first: frame point, idx}
  var busy = false;
  var view = { z: 1, ox: 0, oy: 0, follow: false };   // zoom, the frame point at the canvas's top
                                                      // left; `follow`: `x` zooms to each box
  var hideBoxes = false;               // `h` held
  var showTags = true;                 // `l` toggles
  var spotlight = pref('spotlight', true);   // `v`: veil the frame and dim the other boxes
  var wideSide = pref('wide', false);        // a larger close-up (wider side panel)
  var hoverIdx = -1;                   // the box under the pointer (its tag shows in full)
  var classGrid = false;               // the decision bar's "It's a..." class buttons open
  var dismissedOn = false;             // `i`: this frame's dismissed boxes are shown
  var spaceDown = false;               // Space held: left-drag pans
  var overCanvas = false;
  var guard = null;                    // {frame, count}: the save that warned about review boxes
  var emptyBackup = null;              // {frame, boxes}: the boxes before "Confirmed empty"
  var upcoming = [];                   // the next frames of the queue (prefetch)
  var prefetchImg = null;
  var seenLabels = [];                 // classes the queue saw (Class filter)
  var lastNudge = null;                // {i, t}: arrow nudges in a row share one undo step
  var labelPicker = null, showPicker = null;
  var backStack = [], fwdStack = [];   // frames left by Back (`p`) / by moving on, for `p` and `n`
  var missingAsked = null;             // the frame "Anything missing?" was asked on
  var askMissing = pref('askMissing', true);
  var frameStart = 0;                  // when the frame was loaded (session pace)
  var session = {
    start: Date.now(), frames: [],     // per saved frame {s, acc, rej}
    breakAt: Date.now() + 45 * 60000   // the next break note
  };
  // View-only image setting: `mode` auto (bright + equalised on night / IR frames, else off), off,
  // bright (levels stretched, lifted), eq (histogram equalised); `gamma` on top ([ / ]).
  var imgView = { mode: 'auto', gamma: 1 };
  try {
    var iv = JSON.parse(window.localStorage.getItem('bop.review.view') || 'null');
    if (iv && /^(auto|off|bright|eq)$/.test(iv.mode)) imgView.mode = iv.mode;
    if (iv && typeof iv.gamma === 'number' && iv.gamma >= 0.4 && iv.gamma <= 2.5) imgView.gamma = iv.gamma;
  } catch (e) { /* the defaults */ }
  var viewCache = null;                // {img, key, canvas}: the adjusted frame
  var dups = { groups: [], of: {}, overlaps: [], near: {} };   // dupGroups, as of the last render
  var suggestions = [];                // liveSuggestions, as of the last render
  var refineCur = null;                // R.refineCurrent ({l, k, n}), as of the last render
  var refinePutOff = [];               // keys (R.refineKey) put off with Later on this visit

  var COLORS = {
    pending: '#e8d000', review: '#ffa000', unlikely: '#ffa000', likely: '#14d6b0', hint: '#f08c00', accepted: '#2e9e4f',
    added: '#1f6fe0', rejected: '#e53935', sugDrop: '#ff3df0', sugKeep: '#00e5ff', ghost: '#b26bff'
  };
  var MAX_ZOOM = 8;
  var HANDLE_CSS = 5;   // half size of a resize handle, CSS pixels
  var CLICK_CSS = 4;    // a pointer that moves less than this is a click
  var TAP_CSS = 10;     // the same for a finger, which is never still
  var TAP_NEAR_CSS = 14; // a tap this close to a box selects it (tiny boxes are smaller than a finger)
  var MIN_BOX = 3;      // frame pixels
  var DISMISS_SCORE = 0.15;   // training/autolabel/triage.rs

  // The status line's keys, in the words of the buttons (no keyboard on a touch screen).
  var TOUCH_WORDS = [
    [' (z undoes)', ' (Undo takes it back)'], [': s saves.', ': Save & next saves.'],
    ['a accepts, r rejects, Shift+R rejects the rest', 'answer each, or Reject the rest'],
    ['a keeps, r rejects, or save again', 'answer each, or save again'], [' (a keep / accept, r reject)', ''],
    ['j selects one', 'tap one'], ['click one, then a accepts it, r rejects it; i puts them away', 'tap one, then answer it'],
    ['n returns, s saves it and returns', 'Next returns, Save & next saves it and returns'],
    ['Shift+U clears every stack', 'Reject stacked dups (under the question) clears every stack'],
    ['Shift+A accepts every proposal', 'Accept proposals accepts every proposal']
  ];

  function say(text, isError) {
    if (text && noHover.matches) TOUCH_WORDS.forEach(function (w) { text = text.replace(w[0], w[1]); });
    msg.textContent = text || '';
    msg.className = 'form-msg' + (isError ? ' error-text' : '');
  }

  // An error that keeps the answer: `status`, `conflict` and the JSON body (`body`).
  function api(url, opts) {
    return fetch(url, opts).then(function (r) {
      return r.json().catch(function () { return null; }).then(function (j) {
        if (!r.ok || !j || j.success === false) {
          var err = new Error((j && j.message) || ('HTTP ' + r.status));
          err.status = r.status;
          err.body = j;
          err.conflict = r.status === 409 || !!(j && j.conflict);
          throw err;
        }
        return j;
      });
    });
  }

  function postJson(url, body) {
    return api(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(body)
    });
  }

  function filterParams() {
    var p = new URLSearchParams();
    var auto = filtersForm.elements.auto ? filtersForm.elements.auto.value : '';
    // `refine_kind` only with `auto=refine` (R.sendsFilter).
    Array.prototype.forEach.call(filtersForm.elements, function (el) {
      if (el.name && R.sendsFilter(el.name, el.value, auto)) p.set(el.name, el.value);
    });
    return p;
  }

  // ---- state helpers ------------------------------------------------------------------------

  function boxes() { return cur ? cur.labels.boxes : []; }

  function snapshot() {
    if (!cur) return;
    undo.push(JSON.stringify({ labels: cur.labels, selected: selected }));
    if (undo.length > 200) undo.shift();
    dirty = true;
  }

  // A pending model box saving rejects unless accepted (suppressed, below threshold, ...).
  function isHint(b) { return b.state === 'pending' && resolved(b) === 'rejected'; }

  // A pending box the auto-labeler flagged: triage `review`, or it names another class (never
  // one of an unwanted class: it is not shown).
  function isReview(b) {
    return b.state === 'pending' && !isUnwanted(b) &&
      (b.triage === 'review' || (!!b.auto_label && b.auto_label !== b.label));
  }

  // A flagged box a save would make a label: it needs a decision first. Twin of
  // `labels::needs_decision`, which refuses the save.
  function mustDecide(b) { return isReview(b) && resolved(b) === 'accepted'; }

  // Set aside by a labelers run for its score or its spot, nobody judged it
  // (labels::LabelBox::is_dismissed).
  function isDismissed(b) { return b.state === 'rejected' && isDismissal(b.reason); }

  // Why a dismissed box is hidden, for the card and the box row.
  function hiddenWhy(b) {
    if (b.reason !== 'spot_dismissed') return 'hidden: score below ' + DISMISS_SCORE;
    var n = b.spot_rejections;
    return 'hidden: you rejected this spot ' + (n ? n + ' time' + (n === 1 ? '' : 's') : 'before');
  }

  // Dismissed and not shown just now.
  function isPutAway(b) { return isDismissed(b) && !dismissedOn; }

  // Not drawn, listed or stepped through.
  function isOff(b) { return isUnwanted(b) || isPutAway(b); }

  function dismissedCount() {
    return boxes().filter(function (b) { return isDismissed(b) && !isUnwanted(b); }).length;
  }

  // A pending box the labelers think is real (`likely`: a score and a VLM yes).
  function isLikely(b) { return b.state === 'pending' && !!b.likely && !isUnwanted(b); }

  // A flagged auto box on a frame a run sorted into tiers. Files of older runs have no tiers, so
  // their flagged boxes keep their full look.
  function isUnlikely(b) {
    return isReview(b) && !b.likely && b.source === 'auto' &&
      boxes().some(function (x) { return x.likely || x.audit || isDismissed(x); });
  }

  // What a box shows as: the state, with pending split by tier, review, what saving makes of it.
  function look(b) {
    if (b.state === 'pending') {
      if (isLikely(b)) return 'likely';
      if (isReview(b)) return isUnlikely(b) ? 'unlikely' : 'review';
      return isHint(b) ? 'hint' : 'pending';
    }
    return b.state;
  }

  // The mark before a box's number: likely, or flagged.
  function markOf(b) { return isLikely(b) ? '★ ' : (isReview(b) ? '? ' : ''); }

  // The box `d` (1 or -1) steps from the selected one, wrapping, past the unwanted boxes (not
  // shown); -1 when none is shown.
  function stepShown(d) {
    var bs = boxes();
    for (var k = 1; k <= bs.length; k++) {
      var j = ((selected < 0 && d < 0 ? 0 : selected) + d * k + bs.length * 2) % bs.length;
      if (!isOff(bs[j])) return j;
    }
    return -1;
  }

  function reviewCount() { return boxes().filter(isReview).length; }

  // The first flagged box, a likely one before the others.
  function firstReview() {
    var bs = boxes(), i;
    for (i = 0; i < bs.length; i++) if (isLikely(bs[i]) && isReview(bs[i])) return i;
    for (i = 0; i < bs.length; i++) if (isReview(bs[i])) return i;
    return -1;
  }

  // A decided box (accepted, rejected, drawn) not looked at on this visit of the frame: the
  // check pass steps through them once nothing is pending. `_checked` is page-only (not saved).
  // A dismissed box only while shown (isShown): nobody asked to judge it.
  function isUnchecked(b) {
    return b.state !== 'pending' && !b._checked && isShown(b);
  }

  function uncheckedCount() { return boxes().filter(isUnchecked).length; }

  function undecidedCount() {
    return boxes().filter(function (b) { return b.state === 'pending' && !isUnwanted(b); }).length;
  }

  // The box to look at after `i` (wrapping; -1 for the first): the next likely box, else the next
  // review box, else the next pending one, else (the check pass) the first unchecked decided one,
  // else -1.
  function nextUndecided(i) {
    var bs = boxes();
    var tests = [isLikely, isReview, function (b) { return b.state === 'pending' && !isUnwanted(b); }];
    for (var t = 0; t < tests.length; t++) {
      for (var k = 1; k <= bs.length; k++) {
        var j = (i + k + bs.length) % bs.length;
        if (j !== i && tests[t](bs[j])) return j;
      }
    }
    for (var c = 0; c < bs.length; c++) if (c !== i && isUnchecked(bs[c])) return c;
    return -1;
  }

  // Where the frame's review stands, after a decision moved the selection on.
  function sayProgress() {
    // One column: the bar at the screen's foot says it (a status line under the card that changes
    // with every answer is one more thing for the page to jump to).
    if (deciding()) { say(''); return; }
    var u = undecidedCount(), c = uncheckedCount();
    if (u) say(u + ' box(es) still undecided.');
    else if (c) say('Check pass: ' + c + ' accepted / rejected box(es) left to double-check (a keep / accept, r reject).');
    else say('Every box decided and checked: s saves.');
  }

  function fmtScore(s) { return (typeof s === 'number') ? s.toFixed(2) : ''; }

  // The classes Review offers: the list minus the hidden ones.
  function visible() { return classes.filter(function (c) { return hiddenClasses.indexOf(c) < 0; }); }

  // A "don't want" class (collect.hidden_classes; case-insensitive).
  function isHiddenClass(label) { return R.isHiddenClass(label, hiddenClasses); }

  // A box of an unwanted class (R.isUnwanted): not drawn, listed or stepped through.
  function isUnwanted(b) { return R.isUnwanted(b, classes, hiddenClasses); }

  // "Show boxes of": a box of another class is dimmed (display only; saving keeps every box).
  function isShown(b) {
    return !isOff(b) && (!shownLabels.length || shownLabels.indexOf(b.label) >= 0);
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function round1(v) { return Math.round(v * 10) / 10; }

  // Stacked and overlapping boxes (R.dupGroups). `list`: other boxes than the frame's own.
  function dupGroups(list) { return R.dupGroups(list || boxes(), classes, hiddenClasses); }

  // The box of `i`'s stack it is compared with: the keeper, or (for the keeper) the most
  // overlapping other one. Outside a stack, the box it overlaps most.
  function dupPartner(i) {
    var g = dups.of[i];
    if (!g) return dups.near[i] != null ? dups.near[i] : -1;
    if (g.keeper !== i) return g.keeper;
    var bs = boxes(), best = -1, bi = -1;
    g.members.forEach(function (k) {
      if (k === i) return;
      var v = iou(bs[i].bbox, bs[k].bbox);
      if (v > bi) { bi = v; best = k; }
    });
    return best;
  }

  // ---- overlap suggestions ------------------------------------------------------------------

  // The suggestions a reviewer can act on now (R.liveSuggestions). Reads `dups`.
  function liveSuggestions() { return R.liveSuggestions(cur, dups); }

  // The suggestion on screen: the one about the selected box, else the strongest.
  function shownSuggestion() {
    return suggestions.filter(function (g) { return g.a === selected || g.b === selected; })[0] || suggestions[0] || null;
  }

  // The box to open a suggestion on: the one it would drop, else its first.
  function suggestionBox(g) { return g.drop >= 0 ? g.drop : g.a; }

  // 'drop', 'keep' or '' for box `i` under the suggestion on screen.
  function suggestionRole(i) {
    var g = shownSuggestion();
    if (!g) return '';
    if (i === g.drop) return 'drop';
    return i === g.a || i === g.b ? 'keep' : '';
  }

  // `y`: the suggestion on screen. A drop rejects the box as a duplicate and keeps the other through
  // the save; two objects keeps both (the save then records `overlaps_checked`). One undo step.
  // The rule is R.acceptOverlap.
  function acceptOverlap() {
    if (!cur) return;
    dups = dupGroups();
    suggestions = liveSuggestions();
    var g = shownSuggestion();
    if (!g) { say('No overlap suggestion on this frame.'); return; }
    snapshot();
    R.acceptOverlap(cur.labels, g);
    var text = g.drop >= 0
      ? 'Box ' + (g.drop + 1) + ' rejected as a duplicate of box ' + (g.keep + 1) + ' (z undoes).'
      : 'Boxes ' + (g.a + 1) + ' and ' + (g.b + 1) + ' kept as two objects (z undoes).';
    dups = dupGroups();
    suggestions = liveSuggestions();
    // After the last one, a dataset check suggestion of the frame comes up (if any).
    refineCur = suggestions.length ? null : currentRefine();
    if (refineCur) focusRefine();
    else {
      var next = suggestions.length ? suggestionBox(suggestions[0]) : nextUndecided(-1);
      select(next >= 0 ? next : (g.drop >= 0 ? g.keep : g.a), true);
    }
    say(text + (suggestions.length ? ' ' + suggestions.length + ' more suggestion(s) on this frame.' : ''));
  }

  // The decision bar's block: what is suggested, why, a warning in the weak band, the key.
  function suggestionBlock() {
    var g = shownSuggestion();
    if (!g) return null;
    var s = g.s, box = el('div', 'rv-q-sug');
    box.appendChild(el('div', 'rv-q-sug-head', g.drop >= 0
      ? 'Suggested: drop box ' + (g.drop + 1) + ', keep box ' + (g.keep + 1) + '.'
      : 'Suggested: two objects, keep boxes ' + (g.a + 1) + ' and ' + (g.b + 1) + '.'));
    if (s.why) box.appendChild(el('div', 'muted', s.why + '.'));
    if (s.low_confidence) box.appendChild(el('div', 'rv-chip warn', 'Lower confidence: the closer look gave no answer, only the detectors count.'));
    if (s.weak) {
      box.appendChild(el('div', 'rv-chip bad', 'Look closely: something (' + fmtScore(s.other_score) +
        ') was seen on the box to drop. 6 of 17 such drops removed a real object.'));
    }
    var row = el('div', 'rv-q-btns');
    row.appendChild(decideButton(g.drop >= 0 ? 'Accept: drop box ' + (g.drop + 1) : 'Accept: keep both', 'y', 'ok', acceptOverlap));
    if (suggestions.length > 1) row.appendChild(el('span', 'muted', (suggestions.length - 1) + ' more on this frame'));
    box.appendChild(row);
    return box;
  }

  // ---- the dataset check's suggestions (labels.refine) --------------------------------------
  //
  // One is current at a time (R.refineCurrent): strongest first, and only once no overlap
  // suggestion is on screen (those keep `y`). Accept (`y`) and No (`w`) are edits like any other:
  // one undo step, kept in a draft, saved with the frame by the ordinary save. Anything else the
  // reviewer does leaves the suggestion open; an edit that makes it no longer apply (a rejected,
  // moved, too-small or unsure box) ends it by itself (R.liveRefine on the frame as it is now).

  function liveRefine() { return cur ? R.liveRefine(cur.labels, classes, hiddenClasses) : []; }

  // The current suggestion, worked out again (reads `suggestions`).
  function currentRefine() { return R.refineCurrent(liveRefine(), suggestions.length > 0, refinePutOff); }

  // What the frame shows for the current suggestion (R.refineFocus), null for none.
  function refineShown() { return cur && refineCur ? R.refineFocus(refineCur.l, boxes()) : null; }

  // The proposed box of the current add / tighten, as a box the drawing and the close-up take.
  function ghostBox() {
    var f = refineShown();
    if (!f || !f.ghost) return null;
    var t = boxes()[f.mark];
    return { bbox: f.ghost, label: refineCur.l.label || (t ? t.label : ''), score: refineCur.l.score, state: 'ghost' };
  }

  // What the view frames: the selected box; the current suggestion's part of the frame when its
  // box is selected (a tighter box: both boxes), or when nothing is selected and it adds a box.
  function viewTarget() {
    var b = boxes()[selected], f = refineShown();
    if (f && f.focus && (f.mark >= 0 ? selected === f.mark : !b)) return { bbox: f.focus };
    return b || null;
  }

  // The current suggestion's kind for box `i` when it is about that box, else ''.
  function refineRole(i) {
    var f = refineShown();
    return f && f.mark >= 0 && f.mark === i ? refineCur.l.kind : '';
  }

  // Bring the current suggestion up: select its box (nothing for an add) and frame it the way a
  // selected box is framed (zoomed while following, else panned into view).
  function focusRefine() {
    var f = refineShown();
    if (!f) return;
    if (f.mark >= 0) { select(f.mark, true); return; }
    selected = -1;
    if (oneColumn.matches) view.follow = pref('follow', true);
    if (f.focus) {
      var g = { bbox: f.focus };
      if (view.follow) zoomTo(g); else panTo(g);
    }
    refresh();
    revealSelected(true);
  }

  // Whether the current suggestion is the one in view (its box selected, or for an add none).
  function refineInView() {
    var f = refineShown();
    return !!f && (f.mark >= 0 ? selected === f.mark : selected < 0);
  }

  // What key `y` or `w` acts on now (R.suggestionKey); `repeat`: the key is held down.
  function suggestionRoute(key, repeat) {
    if (!cur) return R.suggestionKey(key, false, null, repeat);
    dups = dupGroups();
    suggestions = liveSuggestions();
    return R.suggestionKey(key, suggestions.length > 0, currentRefine(), repeat);
  }

  // `y` (yes) / `w` (no) on the current suggestion, one undo step (R.answerRefine). Then the next
  // suggestion comes up; after the last, the next box to look at. An accepted add leaves its new
  // box selected, an ordinary box to adjust or relabel.
  function answerRefine(yes) {
    if (!cur) return;
    dups = dupGroups();
    suggestions = liveSuggestions();
    var c = currentRefine();
    if (!c) { say('No dataset check suggestion on this frame.'); return; }
    var l = c.l, card = R.refineCard(l, boxes(), noHover.matches);
    var was = dirty;
    snapshot();
    var r = R.answerRefine(cur.labels, l, yes, classes, hiddenClasses);
    if (!r.ok) {
      undo.pop();
      dirty = was;
      refresh();
      say('That suggestion no longer applies.');
      return;
    }
    lastNudge = null;
    dups = dupGroups();
    suggestions = liveSuggestions();
    refineCur = currentRefine();
    if (yes && l.kind === 'add') select(r.box, true);
    else if (refineCur) focusRefine();
    else {
      var next = nextUndecided(-1);
      select(next >= 0 ? next : r.box, true);
    }
    var text = (card ? (yes ? card.said_yes : card.said_no) : 'Answered') + ' (z undoes).';
    say(text + (refineCur ? ' ' + refineCur.n + ' more suggestion(s) on this frame.' : ' No suggestion left on this frame: s saves.'));
  }

  // Later: the next suggestion of this frame, this one stays open (after the last: the first).
  function putOffRefine() {
    if (!cur) return;
    refinePutOff = R.refineLater(liveRefine(), suggestions.length > 0, refinePutOff);
    refineCur = currentRefine();
    focusRefine();
  }

  // The card of the current suggestion: what is suggested, why, the score, its place, the
  // answers (in the bar at the screen's foot while it shows, else here).
  function refineBlock(inBar) {
    if (!refineCur) return null;
    var l = refineCur.l, card = R.refineCard(l, boxes(), noHover.matches);
    if (!card) return null;
    var box = el('div', 'rv-q-sug rv-q-refine');
    box.appendChild(el('div', 'rv-q-sug-head', card.head));
    if (l.why) box.appendChild(el('div', 'muted', l.why));
    box.appendChild(el('div', 'muted', ['Dataset check', card.score, refineCur.k + ' of ' + refineCur.n + ' on this frame']
      .filter(Boolean).join(' · ')));
    if (card.note) box.appendChild(el('div', 'rv-q-refine-note', card.note));
    var row = el('div', 'rv-q-btns');
    var answers = inBar ? barBtns : row;
    answers.appendChild(decideButton(card.yes, 'y', 'ok', function () { answerRefine(true); }));
    answers.appendChild(decideButton(card.no, 'w', card.no_bad ? 'bad' : 'secondary', function () { answerRefine(false); }));
    if (!refineInView()) {
      var show = decideButton('Show it', null, 'secondary', focusRefine);
      show.title = 'Select the box this suggestion is about and bring it into view';
      row.appendChild(show);
    }
    if (refineCur.n > 1) {
      var later = decideButton('Later', null, 'secondary', putOffRefine);
      later.title = 'The next suggestion of this frame; this one stays open';
      row.appendChild(later);
    }
    if (row.childNodes.length) box.appendChild(row);
    return box;
  }

  // A frame with unsaved edits asks before it is left.
  function leaveOk() {
    if (!cur || !dirty) return true;
    return window.confirm('Discard ' + (undo.length || 1) + ' unsaved change(s) on this frame?');
  }

  // ---- view-only image adjustment -------------------------------------------------------------

  function darkScene() {
    var s = cur && cur.record && cur.record.scene;
    return s === 'night' || s === 'ir';
  }

  // The mode in effect on this frame: `auto` is bright + equalised on night / IR frames.
  function imgMode() {
    return imgView.mode === 'auto' ? (darkScene() ? 'eq' : 'off') : imgView.mode;
  }

  var IMG_NAMES = { off: 'off', bright: 'bright', eq: 'bright+eq' };

  function imgLabel() {
    return 'Image ' + (imgView.mode === 'auto' ? 'auto: ' + IMG_NAMES[imgMode()] : IMG_NAMES[imgView.mode]) +
      (imgView.gamma !== 1 ? ' γ' + imgView.gamma.toFixed(2) : '');
  }

  function setImgView(mode, gamma) {
    imgView.mode = mode;
    imgView.gamma = gamma;
    try { window.localStorage.setItem('bop.review.view', JSON.stringify(imgView)); } catch (e) { /* not kept */ }
    say(imgLabel() + ' (view only: the frame is not changed).');
    draw();
  }

  // `b`: auto -> off -> bright -> bright + equalised -> auto.
  function cycleImg() {
    var order = ['auto', 'off', 'bright', 'eq'];
    setImgView(order[(order.indexOf(imgView.mode) + 1) % order.length], imgView.gamma);
  }

  // `[` / `]`: darker / brighter (gamma); back at 1 snaps to it.
  function stepGamma(d) {
    var g = clamp(imgView.gamma * (d > 0 ? 1 / 0.85 : 0.85), 0.4, 2.5);
    if (Math.abs(g - 1) < 0.04) g = 1;
    setImgView(imgView.mode, Math.round(g * 100) / 100);
  }

  // A 256-entry table from the frame's luma histogram: `bright` stretches the levels (1% .. 99.5%)
  // and lifts them, `eq` mostly equalises them; the gamma on top.
  function viewLut(hist, total, mode, gamma) {
    var lut = new Uint8ClampedArray(256), cdf = new Float64Array(256), acc = 0, i;
    for (i = 0; i < 256; i++) { acc += hist[i]; cdf[i] = acc / total; }
    var lo = 0, hi = 255;
    while (lo < 254 && cdf[lo] < 0.01) lo++;
    while (hi > lo + 1 && cdf[hi - 1] >= 0.995) hi--;
    var g = gamma * (mode === 'bright' ? 1.5 : 1);
    for (i = 0; i < 256; i++) {
      var v = i / 255;
      if (mode !== 'off') v = clamp((i - lo) / Math.max(1, hi - lo), 0, 1);
      if (mode === 'eq') v = 0.75 * cdf[i] + 0.25 * v;
      lut[i] = Math.round(255 * Math.pow(v, 1 / g));
    }
    return lut;
  }

  // What the frame and the close-up draw: the frame, or a copy of it through the table (made
  // again only when the frame or the setting changes; the saved frame never changes).
  function viewSource() {
    if (!img || !img.complete || !img.naturalWidth) return null;
    var mode = imgMode();
    if (mode === 'off' && imgView.gamma === 1) return img;
    var key = mode + ':' + imgView.gamma;
    if (viewCache && viewCache.img === img && viewCache.key === key) return viewCache.canvas;
    try {
      var w = img.naturalWidth, h = img.naturalHeight;
      var oc = (viewCache && viewCache.canvas) || document.createElement('canvas');
      oc.width = w; oc.height = h;
      var c = oc.getContext('2d');
      c.drawImage(img, 0, 0);
      var data = c.getImageData(0, 0, w, h), d = data.data, i;
      // The histogram from about 250k pixels.
      var hist = new Float64Array(256), n = 0, step = Math.max(1, Math.floor(d.length / 4 / 250000)) * 4;
      for (i = 0; i < d.length; i += step) { hist[(d[i] * 77 + d[i + 1] * 150 + d[i + 2] * 29) >> 8]++; n++; }
      var lut = viewLut(hist, n || 1, mode, imgView.gamma);
      for (i = 0; i < d.length; i += 4) { d[i] = lut[d[i]]; d[i + 1] = lut[d[i + 1]]; d[i + 2] = lut[d[i + 2]]; }
      c.putImageData(data, 0, 0);
      viewCache = { img: img, key: key, canvas: oc };
      return oc;
    } catch (e) {
      return img;   // a frame the page cannot read back: shown as it is
    }
  }

  // ---- view (zoom / pan) --------------------------------------------------------------------

  // Canvas pixels per CSS pixel.
  function cssScale() {
    var r = canvas.getBoundingClientRect();
    return r.width ? canvas.width / r.width : 1;
  }

  function clampView() {
    view.z = clamp(view.z, 1, MAX_ZOOM);
    if (view.z < 1.001) { view.z = 1; view.ox = 0; view.oy = 0; return; }
    view.ox = clamp(view.ox, 0, canvas.width - canvas.width / view.z);
    view.oy = clamp(view.oy, 0, canvas.height - canvas.height / view.z);
  }

  function fit() {
    view.z = 1; view.ox = 0; view.oy = 0; view.follow = false;
    draw();
  }

  // Zoom so the box `b` with a margin fills about 60% of the view, centred on it.
  function zoomTo(b) {
    if (!b) return;
    var W = canvas.width, H = canvas.height;
    var bw = Math.max(MIN_BOX, b.bbox[2] - b.bbox[0]), bh = Math.max(MIN_BOX, b.bbox[3] - b.bbox[1]);
    view.z = clamp(Math.min(W * 0.6 / bw, H * 0.6 / bh), 1, MAX_ZOOM);
    view.ox = (b.bbox[0] + b.bbox[2]) / 2 - W / view.z / 2;
    view.oy = (b.bbox[1] + b.bbox[3]) / 2 - H / view.z / 2;
    clampView();
  }

  // The touch UI's Zoom to box / Fit toggle reads Fit (and fits) while following the selection,
  // or when there is no box to zoom to.
  function zoomBoxFits() { return view.follow || !viewTarget(); }

  // `x`: zoom to the selected box and keep following the selection (Fit, the wheel or a pan stop it).
  // With a dataset check suggestion on screen, to what viewTarget frames.
  function zoomToSelected() {
    var b = viewTarget();
    if (!b) { say('No box selected: j selects one.'); return; }
    view.follow = true;
    zoomTo(b);
    draw();
  }

  // Zoom by `factor` keeping the canvas point (cx, cy) on the same frame point.
  function zoomAt(cx, cy, factor) {
    var fx = cx / view.z + view.ox, fy = cy / view.z + view.oy;
    view.z = clamp(view.z * factor, 1, MAX_ZOOM);
    view.ox = fx - cx / view.z;
    view.oy = fy - cy / view.z;
    clampView();
    draw();
  }

  // Zoomed: pan so the box `b` is in view (centred when it is not).
  function panTo(b) {
    if (!b || view.z <= 1) return;
    var vw = canvas.width / view.z, vh = canvas.height / view.z;
    var bb = b.bbox;
    if (bb[0] >= view.ox && bb[2] <= view.ox + vw && bb[1] >= view.oy && bb[3] <= view.oy + vh) return;
    view.ox = (bb[0] + bb[2]) / 2 - vw / 2;
    view.oy = (bb[1] + bb[3]) / 2 - vh / 2;
    clampView();
  }

  function updateViewUi() {
    var zoomed = view.z > 1;
    var sb = viewTarget();
    if (touchUi.matches) {
      // One toggle: Fit while following the selection (or with no box to zoom to), else Zoom to box.
      if (fitBtn) fitBtn.hidden = true;
      if (zoomBoxBtn) {
        zoomBoxBtn.textContent = zoomBoxFits() ? 'Fit' : 'Zoom to box';
        zoomBoxBtn.hidden = !sb && !zoomed;
        // Its text names what a tap does; no pressed state (a filled "Fit" reads as the main action).
        zoomBoxBtn.removeAttribute('aria-pressed');
      }
    } else {
      if (fitBtn) fitBtn.hidden = !zoomed;
      if (zoomBoxBtn) {
        zoomBoxBtn.textContent = 'Zoom to box';
        zoomBoxBtn.hidden = !sb;
        zoomBoxBtn.removeAttribute('aria-pressed');
      }
    }
    if (zoomText) zoomText.textContent = zoomed ? Math.round(view.z * 100) + '%' : '';
    // The touch UI has the image setting in the View menu.
    if (imgBtn) imgBtn.hidden = touchUi.matches;
    if (imgBtn) {
      imgBtn.textContent = imgLabel();
      imgBtn.title = touchUi.matches
        ? 'View only: cycles auto / off / bright / bright + equalised; View has Brighter, Darker and Reset image'
        : 'View only: b cycles auto / off / bright / bright + equalised, [ and ] darker / brighter, Shift+0 resets';
    }
    touchViewUi();
  }

  // ---- drawing ------------------------------------------------------------------------------

  function lineWidth() { return Math.max(2, Math.round(canvas.width / 400)); }

  var media = function (q) { return window.matchMedia ? window.matchMedia(q) : { matches: false }; };
  // The one-column page (same width as the stylesheet); small canvas or a finger.
  var oneColumn = media('(max-width: 900px)');
  var touchUi = media('(max-width: 900px), (pointer: coarse)');
  // No hover: the keys and "the pointer over a box" mean nothing, the answers name the buttons.
  var noHover = media('(hover: none)');
  // A finger is the main pointer (phones, tablets).
  var coarse = media('(pointer: coarse)');
  // One column: the frame below the overview is the editor, so it follows the selected box unless
  // the viewer chose Fit (`follow`, per viewer; loadFrame takes it again on each frame).
  if (oneColumn.matches) view.follow = pref('follow', true);

  // The box's shorter side as drawn, CSS pixels.
  function boxSideCss(bb) {
    return Math.min(bb[2] - bb[0], bb[3] - bb[1]) * view.z / cssScale();
  }

  // Half size of a handle in CSS pixels. It follows the box, so a small box keeps its object in
  // view: a sixth of the shorter side, 2 to 5 px (any pointer; the desktop hit area stays 7 px).
  function handleCss(bb) {
    return clamp(boxSideCss(bb) / 6, 2, HANDLE_CSS);
  }

  function handlePoints(bb) {
    var cx = (bb[0] + bb[2]) / 2, cy = (bb[1] + bb[3]) / 2;
    var corners = [['nw', bb[0], bb[1]], ['ne', bb[2], bb[1]], ['se', bb[2], bb[3]], ['sw', bb[0], bb[3]]];
    var shorter = boxSideCss(bb);
    // Small boxes: the side handles go first, a box under 10 CSS px has none (a drag moves it).
    if (shorter < 10) return [];
    if (shorter < 24) return corners;
    return [corners[0], ['n', cx, bb[1]], corners[1], ['e', bb[2], cy],
      corners[2], ['s', cx, bb[3]], corners[3], ['w', bb[0], cy]];
  }

  function tagText(b, i) {
    var l = look(b);
    return markOf(b) + (b.difficult ? '?! ' : '') + (i + 1) + ' ' + b.label +
      (b.score != null ? ' ' + fmtScore(b.score) : '') +
      (b.source === 'auto' && b.triage === 'accept' && b.state !== 'rejected' ? ' ✓' : '');
  }

  // The tag's lead for the box the current dataset check suggestion is about (refineRole).
  function refineTag(kind) {
    if (kind === 'verify') return 'check? ';
    if (kind === 'relabel') return '→ ' + refineCur.l.label + '? ';
    if (kind === 'tighten') return 'tighten? ';
    return '';
  }

  // Spotlight on: a selected box is the only bright one (veil over the frame, the others dim).
  function spotlit() { return spotlight && !!boxes()[selected]; }

  function draw() {
    var W = canvas.width, H = canvas.height;
    clampView();
    updateViewUi();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    var z = view.z;
    ctx.setTransform(z, 0, 0, z, -view.ox * z, -view.oy * z);
    var src = viewSource();
    if (src) ctx.drawImage(src, 0, 0, W, H);
    drawCrop();
    drawOverview();
    if (!cur || hideBoxes) { drawTapMarker(); return; }
    var lw = lineWidth() / z;
    // Tags keep one size on screen at any zoom; on the touch UI they grow with the zoom up to
    // 11 CSS px (a large frame on a phone has them tiny at zoom 1).
    var base = Math.max(12, Math.round(W / 70));
    var fpx = touchUi.matches ? Math.max(base, Math.min(11 * cssScale(), base * z)) : base;
    var font = fpx / z;
    var pad = 4 * fpx / base / z;
    var th = font + pad;
    // The part of the frame in view: tags are placed against it.
    var vx0 = view.ox, vy0 = view.oy, vx1 = view.ox + W / z, vy1 = view.oy + H / z;
    ctx.font = font + 'px system-ui, sans-serif';
    ctx.textBaseline = 'bottom';
    var spot = spotlit();
    var sb = boxes()[selected];
    if (spot) {
      // The veil, with the selected box cut out.
      ctx.save();
      ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
      ctx.beginPath();
      ctx.rect(0, 0, W, H);
      ctx.rect(sb.bbox[0], sb.bbox[1], sb.bbox[2] - sb.bbox[0], sb.bbox[3] - sb.bbox[1]);
      ctx.fill('evenodd');
      ctx.restore();
    }

    function drawBox(b, i) {
      var l = look(b);
      var x = b.bbox[0], y = b.bbox[1], w = b.bbox[2] - b.bbox[0], h = b.bbox[3] - b.bbox[1];
      var color = COLORS[l] || '#ffffff';
      var dashed = l === 'rejected' || isHint(b);
      var sel = i === selected;
      var role = suggestionRole(i);
      // The box the current dataset check suggestion is about (never while an overlap one shows).
      var rrole = role ? '' : refineRole(i);
      var dim = spot && !sel && !role && !rrole;
      ctx.save();
      var shown = isShown(b);
      var strong = l === 'review' || l === 'likely';
      // Shown dismissed boxes and the unlikely ones sit back: the eye goes to the likely ones.
      var quiet = isDismissed(b) ? 0.85 : (l === 'unlikely' ? 0.6 : 1);
      ctx.globalAlpha = (l === 'rejected' ? 0.7 : 1) * quiet * (shown ? 1 : 0.2) * (dim ? 0.35 : 1);
      if (sel) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = lw * 3.5;
        ctx.strokeRect(x, y, w, h);
      }
      // Dashed: rejected or rejected on save; dotted: an unsure box.
      ctx.setLineDash(dashed ? [lw * 4, lw * 3] : (b.difficult ? [lw, lw * 1.5] : []));
      ctx.lineWidth = (sel ? 2 : 1) * (strong ? 1.5 : 1) * lw;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
      if (strong) {
        // A dark inner line: the flagged boxes read as outlined.
        ctx.setLineDash([]);
        ctx.lineWidth = lw * 0.6;
        ctx.strokeStyle = '#000000';
        var inset = (sel ? 2.25 : 1.1) * lw;
        if (w > 2 * inset && h > 2 * inset) ctx.strokeRect(x + inset, y + inset, w - 2 * inset, h - 2 * inset);
        ctx.strokeStyle = color;
      }
      if (role) {
        // The pair of the suggestion on screen, outside the box: dashed around the box it drops.
        var o = 2.5 * lw;
        ctx.setLineDash(role === 'drop' ? [lw * 3, lw * 2] : []);
        ctx.lineWidth = lw * 1.5;
        ctx.strokeStyle = role === 'drop' ? COLORS.sugDrop : COLORS.sugKeep;
        ctx.strokeRect(x - o, y - o, w + 2 * o, h + 2 * o);
        ctx.strokeStyle = color;
      }
      if (rrole) {
        // Outside the box, in the suggestion's colour: a solid line on a dark one.
        var ro = 2.5 * lw;
        ctx.setLineDash([]);
        ctx.lineWidth = lw * 2.5;
        ctx.strokeStyle = '#000000';
        ctx.strokeRect(x - ro, y - ro, w + 2 * ro, h + 2 * ro);
        ctx.lineWidth = lw * 1.5;
        ctx.strokeStyle = COLORS.ghost;
        ctx.strokeRect(x - ro, y - ro, w + 2 * ro, h + 2 * ro);
        ctx.strokeStyle = color;
      }
      if (l === 'rejected') {
        ctx.setLineDash(dashed ? [lw * 4, lw * 3] : []);
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.moveTo(x, y); ctx.lineTo(x + w, y + h);
        ctx.moveTo(x + w, y); ctx.lineTo(x, y + h);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      // Tags: in full for the selected and the hovered box; none on dimmed boxes; a decided box
      // already checked shows its number only.
      var full = sel || i === hoverIdx;
      var tag = showTags && shown && (full || !dim);
      if (tag) {
        if (full) ctx.globalAlpha = 1;
        var text = (role === 'drop' ? 'drop? ' : (role === 'keep' ? 'keep ' : '')) + refineTag(rrole) +
          ((full || b.state === 'pending' || !b._checked) ? tagText(b, i) : String(i + 1));
        var tw = ctx.measureText(text).width + 2 * pad;
        // Above the box; at the view's top below it when there is room; else inside its top
        // (ReviewTouch.tagTop); across, kept in view.
        var top = window.ReviewTouch.tagTop(b.bbox, vy0, vy1, th);
        var tx = clamp(x, vx0, Math.max(vx0, vx1 - tw));
        ctx.fillStyle = color;
        ctx.fillRect(tx, top, tw, th);
        ctx.fillStyle = '#000000';
        ctx.fillText(text, tx + pad, top + th - pad / 2);
      }
      ctx.restore();
    }

    // The selected box last (on top), the hovered one just before it.
    boxes().forEach(function (b, i) {
      if (!isOff(b) && i !== selected && i !== hoverIdx) drawBox(b, i);
    });
    var hb = boxes()[hoverIdx];
    if (hb && hoverIdx !== selected && !isOff(hb)) drawBox(hb, hoverIdx);
    if (sb && !isOff(sb)) drawBox(sb, selected);
    // The proposed box of the current dataset check suggestion (add, tighten): dashed in its own
    // colour over a dark dash, a light fill, its own tag; no box of the frame looks like it.
    var gb = ghostBox();
    if (gb) {
      var gx = gb.bbox[0], gy = gb.bbox[1], gw = gb.bbox[2] - gb.bbox[0], gh = gb.bbox[3] - gb.bbox[1];
      ctx.save();
      ctx.fillStyle = 'rgba(178, 107, 255, 0.16)';
      ctx.fillRect(gx, gy, gw, gh);
      ctx.setLineDash([lw * 5, lw * 3]);
      ctx.lineWidth = lw * 3;
      ctx.strokeStyle = '#000000';
      ctx.strokeRect(gx, gy, gw, gh);
      ctx.lineWidth = lw * 1.75;
      ctx.strokeStyle = COLORS.ghost;
      ctx.strokeRect(gx, gy, gw, gh);
      ctx.setLineDash([]);
      if (showTags) {
        var gt = (refineCur.l.kind === 'add' ? 'add? ' : 'tighter? ') + gb.label +
          (gb.score != null ? ' ' + fmtScore(gb.score) : '');
        var gtw = ctx.measureText(gt).width + 2 * pad;
        var gtop = window.ReviewTouch.tagTop(gb.bbox, vy0, vy1, th);
        var gtx = clamp(gx, vx0, Math.max(vx0, vx1 - gtw));
        ctx.fillStyle = COLORS.ghost;
        ctx.fillRect(gtx, gtop, gtw, th);
        ctx.fillStyle = '#000000';
        ctx.fillText(gt, gtx + pad, gtop + th - pad / 2);
      }
      ctx.restore();
    }
    if (sb && !(gesture && gesture.mode === 'draw' && gesture.moved)) {
      var hs = handleCss(sb.bbox) * cssScale() / z;
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1 / z;
      handlePoints(sb.bbox).forEach(function (p) {
        ctx.fillRect(p[1] - hs, p[2] - hs, 2 * hs, 2 * hs);
        ctx.strokeRect(p[1] - hs, p[2] - hs, 2 * hs, 2 * hs);
      });
      ctx.restore();
    }
    if (gesture && gesture.mode === 'draw' && gesture.moved) {
      ctx.save();
      ctx.lineWidth = lw;
      ctx.setLineDash([lw * 3, lw * 2]);
      ctx.strokeStyle = COLORS.added;
      var r = norm(gesture);
      ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      ctx.restore();
    }
    drawTapMarker();
  }

  function norm(d) {
    return [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
  }

  // The window height the close-up is sized by. A finger screen keeps the one it had until the
  // width or the orientation changes: iOS fires height-only resizes as its toolbars collapse and
  // grow while the page scrolls, and a close-up that followed them would make the page jump.
  var cropWin = null;   // {w, h, landscape}
  function cropWindowHeight() {
    var w = window.innerWidth, h = window.innerHeight, land = w > h;
    if (!coarse.matches || !cropWin || cropWin.w !== w || cropWin.landscape !== land) cropWin = { w: w, h: h, landscape: land };
    return cropWin.h;
  }

  // The first corner of a two-tap box (Draw, Redraw): a cross in a ring, the same size on screen
  // at any zoom.
  function drawTapMarker() {
    if (!taps || !taps.first) return;
    var z = view.z, k = cssScale() / z, p = taps.first, r = 9 * k;
    ctx.save();
    ctx.setTransform(z, 0, 0, z, -view.ox * z, -view.oy * z);
    ctx.lineWidth = 3 * k;
    ctx.strokeStyle = '#000000';
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
    ctx.moveTo(p[0] - 2 * r, p[1]); ctx.lineTo(p[0] + 2 * r, p[1]);
    ctx.moveTo(p[0], p[1] - 2 * r); ctx.lineTo(p[0], p[1] + 2 * r);
    ctx.stroke();
    ctx.lineWidth = 1.5 * k;
    ctx.strokeStyle = taps.kind === 'redraw' ? '#ffffff' : COLORS.added;
    ctx.stroke();
    ctx.restore();
  }

  // The close-up: the selected box with a 40% margin, enlarged to the panel's width (a box under
  // 120 px at least 2x, at most 8x; the view then centres on the box), from the full-size image.
  // Two columns only: on one column the frame itself is the close-up (it follows the selected box)
  // and the overview over it shows where that is.
  function drawCrop() {
    if (!cropCanvas) return;
    // With no box selected, the proposed box of a dataset check suggestion (an add).
    var gb = ghostBox();
    var b = boxes()[selected] || gb;
    if (!b || !img || !img.complete || !img.naturalWidth || oneColumn.matches) { cropWrap.hidden = true; return; }
    cropWrap.hidden = false;
    var W = canvas.width, H = canvas.height;
    var bw = Math.max(1, b.bbox[2] - b.bbox[0]), bh = Math.max(1, b.bbox[3] - b.bbox[1]);
    var aw = Math.max(120, cropWrap.clientWidth || 300);
    // Its height follows the window: the decision bar and the list stay in view under it.
    var winH = cropWindowHeight();
    var ah = Math.round(Math.max(140, Math.min(wideSide ? 340 : 240, winH * (wideSide ? 0.36 : 0.22))));
    var mw = bw * 1.8, mh = bh * 1.8;
    var scale = Math.min(aw / mw, ah / mh);
    if (Math.max(bw, bh) < 120) scale = Math.max(scale, 2);
    scale = Math.min(scale, 8);
    var cw = Math.min(aw, mw * scale), ch = Math.min(ah, mh * scale);
    var sw = Math.min(W, cw / scale), sh = Math.min(H, ch / scale);
    cw = sw * scale; ch = sh * scale;
    var x0 = clamp((b.bbox[0] + b.bbox[2]) / 2 - sw / 2, 0, W - sw);
    var y0 = clamp((b.bbox[1] + b.bbox[3]) / 2 - sh / 2, 0, H - sh);
    var dpr = window.devicePixelRatio || 1;
    var pw = Math.max(1, Math.round(cw * dpr)), ph = Math.max(1, Math.round(ch * dpr));
    if (cropCanvas.width !== pw) cropCanvas.width = pw;
    if (cropCanvas.height !== ph) cropCanvas.height = ph;
    cropCanvas.style.width = Math.round(cw) + 'px';
    cropCanvas.style.height = Math.round(ch) + 'px';
    var c = cropCanvas.getContext('2d');
    var kx = img.naturalWidth / W, ky = img.naturalHeight / H;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, pw, ph);
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = 'high';
    c.drawImage(viewSource() || img, x0 * kx, y0 * ky, sw * kx, sh * ky, 0, 0, pw, ph);
    // A thin dashed outline just outside the box: the object's edges stay visible.
    var k = scale * dpr, o = 2 * dpr;
    c.lineWidth = 1.5 * dpr;
    c.setLineDash(b.difficult ? [2 * dpr, 3 * dpr] : [5 * dpr, 3 * dpr]);
    c.strokeStyle = COLORS[look(b)] || '#ffffff';
    var rx = (b.bbox[0] - x0) * k - o, ry = (b.bbox[1] - y0) * k - o, rw = bw * k + 2 * o, rh = bh * k + 2 * o;
    c.strokeRect(rx, ry, rw, rh);
    if (gb && gb !== b && refineRole(selected) === 'tighten') {
      // A tighter box proposed for the selected one: dashed in the suggestion's colour.
      c.lineWidth = 2 * dpr;
      c.setLineDash([6 * dpr, 3 * dpr]);
      c.strokeStyle = COLORS.ghost;
      c.strokeRect((gb.bbox[0] - x0) * k, (gb.bbox[1] - y0) * k, (gb.bbox[2] - gb.bbox[0]) * k, (gb.bbox[3] - gb.bbox[1]) * k);
    }
    c.setLineDash([]);
    cropTag(c, b, rx, ry, rh, pw, ph, dpr);
    cropCap.textContent = (b === gb ? 'Proposed box' : 'Box ' + (selected + 1)) + ' (' + b.label + '), ' + Math.round(bw) + ' x ' + Math.round(bh) + ' px, ' +
      (Math.round(scale * 10) / 10) + 'x';
  }

  // The close-up's tag: the class at a glance (bold), the score, the decision so far (? flagged,
  // ✓ accepted / drawn, ✗ rejected, ?! unsure) and the labelers' class when they name another
  // one. Above the outline when there is room, else below it, else inside its top; kept within
  // the close-up.
  function cropTag(c, b, rx, ry, rh, pw, ph, dpr) {
    var l = look(b);
    var mark = b.difficult ? '?! ' : (markOf(b) || (l === 'accepted' || l === 'added' ? '✓ ' : (l === 'rejected' ? '✗ ' : '')));
    var main = mark + b.label;
    var extra = (b.score != null ? ' ' + fmtScore(b.score) : '') +
      (b.auto_label && b.auto_label !== b.label ? ' · labelers: ' + b.auto_label : '');
    var fs = 14 * dpr, pad = 5 * dpr, th = fs + 2 * pad;
    var bold = '600 ' + fs + 'px system-ui, sans-serif', thin = (fs * 0.85) + 'px system-ui, sans-serif';
    c.font = bold;
    var mw = c.measureText(main).width;
    c.font = thin;
    var tw = Math.min(pw, mw + c.measureText(extra).width + 2 * pad);
    var top;
    if (ry - th >= 0) top = ry - th;
    else if (ry + rh + th <= ph) top = ry + rh;
    else top = Math.max(0, ry);
    var tx = clamp(rx, 0, Math.max(0, pw - tw));
    c.fillStyle = COLORS[l] || '#ffffff';
    c.fillRect(tx, top, tw, th);
    c.fillStyle = '#000000';
    c.textBaseline = 'middle';
    c.font = bold;
    c.fillText(main, tx + pad, top + th / 2);
    c.font = thin;
    c.fillText(extra, tx + pad + mw, top + th / 2);
  }

  // The overview (one column, while the frame is zoomed or follows the selection): the whole frame
  // at the overview's CSS width (x devicePixelRatio), every shown box as a thin outline, the
  // selected one thicker on a white line, and a veil outside the part the frame below shows. Held
  // while a gesture runs: the frame would move under the fingers of a pinch that crossed zoom 1.
  var overviewCache = null;   // {img, src, key, w, h, canvas}: the frame at the overview's size
  function drawOverview() {
    if (!overview) return;
    // Following keeps it shown when a box too big to zoom on comes up (zoom 1): the frame and the
    // answers under it stay where they were from one decision to the next.
    var show = !!(oneColumn.matches && cur && (view.z > 1 || view.follow));
    if (gesture) show = !overview.hidden;
    if (overview.hidden === show) overview.hidden = !show;
    if (stage) stage.classList.toggle('rv-ov-on', show);
    if (!show) return;
    var W = canvas.width, H = canvas.height;
    var dpr = window.devicePixelRatio || 1;
    var cssW = overview.getBoundingClientRect().width || overview.clientWidth || 300;
    var pw = Math.max(1, Math.round(cssW * dpr)), ph = Math.max(1, Math.round(pw * H / W));
    if (overview.width !== pw) overview.width = pw;
    if (overview.height !== ph) overview.height = ph;
    var c = overview.getContext('2d');
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, pw, ph);
    var src = viewSource();
    if (src) {
      // Scaled down once per frame, setting and size (a pinch redraws this on every move).
      var vkey = imgMode() + ':' + imgView.gamma;
      var oc0 = overviewCache;
      if (!oc0 || oc0.img !== img || oc0.src !== src || oc0.key !== vkey || oc0.w !== pw || oc0.h !== ph) {
        var oc = (overviewCache && overviewCache.canvas) || document.createElement('canvas');
        oc.width = pw; oc.height = ph;
        var occ = oc.getContext('2d');
        occ.imageSmoothingEnabled = true;
        occ.imageSmoothingQuality = 'high';
        occ.drawImage(src, 0, 0, pw, ph);
        overviewCache = { img: img, src: src, key: vkey, w: pw, h: ph, canvas: oc };
      }
      c.drawImage(overviewCache.canvas, 0, 0);
    }
    var k = pw / W;
    var sb = boxes()[selected];
    if (!hideBoxes) {
      boxes().forEach(function (b, i) {
        if (i === selected || !isShown(b)) return;
        c.lineWidth = 1.25 * dpr;
        c.strokeStyle = COLORS[look(b)] || '#ffffff';
        c.strokeRect(b.bbox[0] * k, b.bbox[1] * k, (b.bbox[2] - b.bbox[0]) * k, (b.bbox[3] - b.bbox[1]) * k);
      });
      if (sb && !isOff(sb)) {
        var r = [sb.bbox[0] * k, sb.bbox[1] * k, (sb.bbox[2] - sb.bbox[0]) * k, (sb.bbox[3] - sb.bbox[1]) * k];
        c.lineWidth = 4.5 * dpr;
        c.strokeStyle = '#ffffff';
        c.strokeRect(r[0], r[1], r[2], r[3]);
        c.lineWidth = 2.25 * dpr;
        c.strokeStyle = COLORS[look(sb)] || '#ffffff';
        c.strokeRect(r[0], r[1], r[2], r[3]);
      }
      var og = ghostBox();
      if (og) {
        c.lineWidth = 2 * dpr;
        c.setLineDash([4 * dpr, 2 * dpr]);
        c.strokeStyle = COLORS.ghost;
        c.strokeRect(og.bbox[0] * k, og.bbox[1] * k, (og.bbox[2] - og.bbox[0]) * k, (og.bbox[3] - og.bbox[1]) * k);
        c.setLineDash([]);
      }
    }
    // The veil outside the part in view, and its outline.
    var vx = view.ox * k, vy = view.oy * k, vw = W / view.z * k, vh = H / view.z * k;
    c.fillStyle = 'rgba(0, 0, 0, 0.45)';
    c.beginPath();
    c.rect(0, 0, pw, ph);
    c.rect(vx, vy, vw, vh);
    c.fill('evenodd');
    c.lineWidth = 1.5 * dpr;
    c.strokeStyle = '#ffffff';
    c.strokeRect(vx, vy, vw, vh);
  }

  // A tap on the overview selects the smallest shown box there (ReviewTouch.pickBox), else moves
  // the frame's view to centre on that point at the same zoom. A drag stays the page's (scroll).
  var overviewDown = null;   // {id, x, y}: the primary pointer down on the overview
  overview.addEventListener('pointerdown', function (ev) {
    overviewDown = ev.button === 0 ? { id: ev.pointerId, x: ev.clientX, y: ev.clientY } : null;
  });
  overview.addEventListener('pointercancel', function () { overviewDown = null; });
  overview.addEventListener('pointerup', function (ev) {
    var d = overviewDown;
    overviewDown = null;
    if (!d || d.id !== ev.pointerId || !cur || overview.hidden) return;
    if (Math.abs(ev.clientX - d.x) > TAP_CSS || Math.abs(ev.clientY - d.y) > TAP_CSS) return;
    var r = overview.getBoundingClientRect();
    if (!r.width || !r.height) return;
    var W = canvas.width, H = canvas.height;
    var x = (ev.clientX - r.left) * W / r.width, y = (ev.clientY - r.top) * H / r.height;
    var idx = [], rects = [];
    if (!hideBoxes) {
      boxes().forEach(function (b, i) { if (isShown(b)) { idx.push(i); rects.push(b.bbox); } });
    }
    // A finger on a small overview: a few CSS px of slack around each box.
    var hit = window.ReviewTouch.pickBox(rects, x, y, 6 * W / r.width);
    if (hit >= 0) { select(idx[hit], false); return; }
    // Unzoomed there is nothing to move to, and ending the follow would fold the overview away.
    if (view.z <= 1) return;
    view.follow = false;
    view.ox = x - W / view.z / 2;
    view.oy = y - H / view.z / 2;
    clampView();
    draw();
  });

  // ---- side panel ---------------------------------------------------------------------------

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  // `cls`: extra classes (`ok` / `bad` colour the decisions).
  function button(text, onClick, cls) {
    var b = el('button', 'small' + (cls ? ' ' + cls : ''), text);
    b.type = 'button';
    b.addEventListener('click', function (ev) { ev.stopPropagation(); onClick(); });
    return b;
  }

  function stateChip(b) {
    if (b.state === 'pending') {
      var r = resolved(b);
      if (mustDecide(b)) return el('span', 'rv-chip warn', 'needs a decision');
      return el('span', 'rv-chip ' + (r === 'rejected' ? 'bad' : 'pend'), 'pending → ' + r + ' on save');
    }
    if (b.state === 'added') return el('span', 'rv-chip drawn', 'drawn');
    return el('span', 'rv-chip ' + (b.state === 'accepted' ? 'ok' : 'bad'), b.state);
  }

  function triageChip(b) {
    if (b.triage === 'accept') return el('span', 'rv-chip ok', 'auto: accept');
    if (b.triage === 'review') {
      var c = el('span', 'rv-chip warn', 'auto: review' + (b.why && b.why.length ? ': ' + b.why.join('; ') : ''));
      return c;
    }
    return null;
  }

  // One muted line: labelers, VLM, SigLIP, the model's score / origin / reason, relabel, moved.
  function evidence(b) {
    var parts = [];
    var names = Object.keys(b.labelers || {});
    if (names.length) {
      parts.push(names.map(function (k) { return k + ' ' + fmtScore(b.labelers[k]); }).join(', ') +
        (b.agree === false && names.length < 2 ? ' (one labeler)' : ''));
    } else if (b.agree === false) parts.push('one labeler');
    if (b.vlm) {
      // `expected: yes`: the yes/no question. Older label files hold a one-word answer.
      if (b.vlm.expected === 'yes') parts.push((b.vlm.model || 'VLM') + (b.vlm.ok ? ': yes' : ' sees no ' + b.label));
      else parts.push((b.vlm.model || 'VLM') + ': ' + (b.vlm.answer || 'nothing') +
        (b.vlm.ok ? '' : (b.vlm.expected ? ' (expected ' + b.vlm.expected + ')' : ' (!)')));
    }
    if (b.species_scores) {
      var best = null;
      Object.keys(b.species_scores).forEach(function (k) { if (!best || b.species_scores[k] > b.species_scores[best]) best = k; });
      if (best) parts.push('SigLIP ' + best + ' ' + fmtScore(b.species_scores[best]));
    }
    if (b.source === 'model') {
      var m = ['model'];
      if (b.score != null && !names.length) m.push(fmtScore(b.score));
      if (b.origin) m.push(b.origin.replace(/_/g, ' '));
      if (b.reason) m.push(b.reason);
      parts.push(m.join(' '));
    } else if (isDismissed(b)) parts.push(hiddenWhy(b));
    else if (b.reason) parts.push(b.reason);
    if (b.model_label && b.model_label !== b.label) parts.push('was ' + b.model_label);
    if (b.orig_bbox) parts.push('moved');
    return parts.join(' · ');
  }

  function classSelect(b, i) {
    var sel = document.createElement('select');
    sel.className = 'bx-label';
    sel.setAttribute('aria-label', 'Class of box ' + (i + 1));
    var names = visible();
    if (names.indexOf(b.label) < 0) names.unshift(b.label);
    names.forEach(function (c) {
      var o = el('option', null, c + (classes.indexOf(c) < 0 ? ' (not a class)' : (hiddenClasses.indexOf(c) >= 0 ? ' (hidden)' : '')));
      o.value = c;
      if (c === b.label) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener('click', function (ev) { ev.stopPropagation(); });
    // Focus goes back to the frame: a focused select would take the next shortcut letter as
    // "jump to the class starting with it" (a relabel) instead of the review key.
    sel.addEventListener('change', function () { relabel(i, sel.value); canvas.focus({ preventScroll: true }); });
    return sel;
  }

  // A short state for the one-line rows.
  function shortState(b) {
    if (b.state === 'pending') {
      var r = resolved(b);
      if (mustDecide(b)) return el('span', 'rv-chip warn', 'decide');
      return el('span', 'rv-chip ' + (r === 'rejected' ? 'bad' : 'pend'), '→ ' + (r === 'rejected' ? 'reject' : 'accept'));
    }
    if (b.state === 'added') return el('span', 'rv-chip drawn', 'drawn');
    return el('span', 'rv-chip ' + (b.state === 'accepted' ? 'ok' : 'bad'), b.state);
  }

  // "Seen here": the reviewer's earlier verdict at the box's spot (repeatOf).
  function repeatChip(r) {
    var text = r.verdict ? 'seen ' + (r.verdict === 'rejected' ? '✗' : '✓') + r.n : 'seen ✓' + r.acc + ' ✗' + r.rej;
    var c = el('span', 'rv-chip' + (r.verdict === 'rejected' ? ' bad' : (r.verdict ? ' ok' : '')), text);
    c.title = 'At this spot on this camera before: accepted ' + r.acc + ', rejected ' + r.rej + ' time(s)' +
      (r.verdict ? '' : ' (mixed: no suggestion)');
    return c;
  }

  // The chip of the box the current dataset check suggestion is about.
  function refineChip(i) {
    var c = el('span', 'rv-chip refine', refineTag(refineRole(i)).trim());
    c.title = 'Dataset check: ' + (refineCur.l.why || 'a suggestion about this box') + ' (the card above answers it)';
    return c;
  }

  // A box that is not selected: one row (number, class, state, score); a click selects it.
  function boxRow(b, i) {
    var li = el('li', 'compact st-' + b.state + ' lk-' + look(b));
    li.appendChild(el('span', 'swatch sw-' + look(b)));
    li.appendChild(el('span', 'bx-num', markOf(b) + String(i + 1)));
    li.appendChild(el('span', 'bx-name', b.label + (b.auto_label && b.auto_label !== b.label ? ' (' + b.auto_label + '?)' : '')));
    li.appendChild(shortState(b));
    if (isLikely(b)) li.appendChild(el('span', 'rv-chip likely', 'likely'));
    if (isDismissed(b)) {
      var hc = el('span', 'rv-chip', 'hidden');
      hc.title = hiddenWhy(b);
      li.appendChild(hc);
    }
    if (b.state !== 'pending' && !b._checked) li.appendChild(el('span', 'rv-chip warn', 'to check'));
    if (dups.of[i] && dups.of[i].keeper !== i) li.appendChild(el('span', 'rv-chip warn', 'dup'));
    else if (dups.near[i] != null) li.appendChild(el('span', 'rv-chip warn', 'overlap'));
    var role = suggestionRole(i);
    if (role) li.appendChild(el('span', 'rv-chip ' + (role === 'drop' ? 'bad' : 'ok'), role === 'drop' ? 'drop?' : 'keep'));
    else if (refineRole(i)) li.appendChild(refineChip(i));
    if (b.difficult) li.appendChild(el('span', 'rv-chip warn', 'unsure'));
    var rp = b.state === 'pending' ? repeatOf(b) : null;
    if (rp) li.appendChild(repeatChip(rp));
    li.appendChild(el('span', 'bx-score muted', fmtScore(b.score)));
    li.addEventListener('click', function () { select(i, oneColumn.matches); });
    return li;
  }

  function boxItem(b, i) {
    if (i !== selected) return boxRow(b, i);
    var li = el('li', 'st-' + b.state + ' lk-' + look(b) + ' sel');
    var head = el('span', null);
    head.appendChild(el('span', 'swatch sw-' + look(b)));
    head.appendChild(document.createTextNode(markOf(b) + String(i + 1) + ' '));
    li.appendChild(head);
    li.appendChild(classSelect(b, i));
    var chips = el('span', 'bx-chips');
    chips.appendChild(stateChip(b));
    if (isLikely(b)) chips.appendChild(el('span', 'rv-chip likely', 'likely'));
    if (isDismissed(b)) chips.appendChild(el('span', 'rv-chip', 'hidden'));
    if (b.state !== 'pending') chips.appendChild(el('span', 'rv-chip' + (b._checked ? ' ok' : ' warn'), b._checked ? 'checked' : 'to check'));
    // The reason of an `auto: review` is in the decision bar above: the chip alone here.
    var t = b.triage === 'review' ? el('span', 'rv-chip warn', 'auto: review') : triageChip(b);
    if (t && b.why && b.why.length) t.title = b.why.join('; ');
    if (t) chips.appendChild(t);
    if (dups.of[i] && dups.of[i].keeper !== i) chips.appendChild(el('span', 'rv-chip warn', 'dup'));
    else if (dups.near[i] != null) chips.appendChild(el('span', 'rv-chip warn', 'overlap'));
    var role = suggestionRole(i);
    if (role) chips.appendChild(el('span', 'rv-chip ' + (role === 'drop' ? 'bad' : 'ok'), role === 'drop' ? 'drop?' : 'keep'));
    else if (refineRole(i)) chips.appendChild(refineChip(i));
    if (b.difficult) chips.appendChild(el('span', 'rv-chip warn', 'unsure'));
    if (b.state === 'rejected' && b.reject_reason) chips.appendChild(el('span', 'rv-chip bad', reasonText(b.reject_reason)));
    li.appendChild(chips);
    var ev = evidence(b);
    if (ev) li.appendChild(el('span', 'bx-note', ev));
    // Accept / reject / keep / delete and "It's a <auto label>" are the decision bar's buttons.
    li.addEventListener('click', function () { select(i, oneColumn.matches); });
    return li;
  }

  function renderSide() {
    boxList.textContent = '';
    var bs = boxes();
    dups = dupGroups();
    suggestions = liveSuggestions();
    refineCur = currentRefine();
    // Likely boxes first, then the other flagged ones, the dismissed ones last; each keeps its
    // number (the one on the canvas).
    var order = [];
    var rank = function (b) { return isLikely(b) ? 0 : (isReview(b) ? 1 : (isDismissed(b) ? 3 : 2)); };
    [0, 1, 2, 3].forEach(function (r) {
      bs.forEach(function (b, i) { if (rank(b) === r) order.push(i); });
    });
    var offScreen = 0, unwanted = 0, putAway = 0, listed = 0;
    order.forEach(function (i) {
      if (isUnwanted(bs[i])) { unwanted++; return; }
      if (isPutAway(bs[i])) { putAway++; return; }
      if (!isShown(bs[i])) { offScreen++; return; }
      boxList.appendChild(boxItem(bs[i], i));
      listed++;
    });
    showListToggle(listed);
    if (offScreen) {
      boxList.appendChild(el('li', 'muted', offScreen + ' box(es) of other classes hidden by "Show boxes of" (still saved).'));
    }
    if (unwanted) {
      boxList.appendChild(el('li', 'muted', unwanted + ' box(es) of classes you do not want (hidden or not in the class list) not shown (kept in the file; pending proposals of them are rejected on save).'));
    }
    if (bs.length === unwanted + putAway && cur) boxList.appendChild(el('li', 'muted', noHover.matches ? 'No box. Turn on Draw to draw one.' : 'No box. Drag on the image to draw one.'));
    emptyBox.checked = !!(cur && cur.labels.empty_confirmed);
    renderTools();
    renderDecision();
  }

  // One column: the fold of the box list, the session line and Panel options (closed by default,
  // per viewer); `n` boxes in the list.
  function showListToggle(n) {
    root.classList.toggle('rv-list-open', listOpen);
    if (!listToggle) return;
    listToggle.textContent = 'Boxes (' + n + ') and options ' + (listOpen ? '▴' : '▾');
    listToggle.setAttribute('aria-expanded', listOpen ? 'true' : 'false');
  }
  if (listToggle) {
    listToggle.addEventListener('click', function () {
      listOpen = !listOpen;
      setPref('list', listOpen);
      renderSide();
      if (listOpen) revealSelected();
    });
  }
  showListToggle(0);

  // "Accept likely (N)" and "Show N hidden" beside the box list.
  function renderTools() {
    boxTools.textContent = '';
    var likely = boxes().filter(isLikely).length, n = dismissedCount();
    if (likely) {
      var acc = decideButton('Accept likely (' + likely + ')', 'c', 'ok rv-tool-likely', acceptLikely);
      acc.title = 'Accept every likely box (★) on this frame; the other boxes stay pending';
      boxTools.appendChild(acc);
    }
    // The bottom bar's copy (phones and tablets).
    if (likelyBar) {
      likelyBar.hidden = !likely;
      likelyBar.textContent = 'Accept likely (' + likely + ')';
      placeAcceptAll();
    }
    if (n) {
      var show = decideButton((dismissedOn ? 'Hide ' : 'Show ') + n + ' hidden', 'i', 'secondary', toggleDismissed);
      show.setAttribute('aria-pressed', dismissedOn ? 'true' : 'false');
      show.title = 'Boxes the auto-labelers set aside, for a score below ' + DISMISS_SCORE + ' (about 5% were real on public sets) or at a spot you keep rejecting. Accept one if it is real.';
      boxTools.appendChild(show);
    }
    boxTools.hidden = !likely && !n;
    boxTools.classList.toggle('rv-only-likely', !!likely && !n);
  }

  // The "Anything missing?" question, with the frame's dismissed boxes when they are off.
  function missingText() {
    var n = dismissedOn ? 0 : dismissedCount();
    if (noHover.matches) {
      return 'Anything missing? Look over the whole frame and draw a missed object with Draw' +
        (n ? ' (' + n + ' hidden: Show them under the question)' : '') + '; Save & next saves.';
    }
    return 'Anything missing? Look over the whole frame and drag to draw a missed object; ' +
      (n ? n + ' hidden: i shows them; ' : '') + 's saves.';
  }

  // The frame's progress: the boxes the reviewer sees, undecided / decided but not checked / done.
  function progress() {
    var shown = boxes().filter(function (b) { return !isUnwanted(b) && isShown(b); });
    var u = shown.filter(function (b) { return b.state === 'pending'; }).length;
    var c = shown.filter(isUnchecked).length;
    return { total: shown.length, undecided: u, unchecked: c, done: shown.length - u - c };
  }

  function kbd(text) { return el('kbd', null, text); }

  // One column: an answer keeps the frame where it was on the screen. Nothing over the frame
  // changes, yet a browser may move the page once the panel under it is rebuilt (scroll anchoring
  // to the rewritten card, iOS bringing the changed status lines into view): the page is put back
  // for a moment after each answer, unless a finger or the wheel moves it, or the page itself
  // scrolls on purpose (revealSelected, revealFrame).
  var frameHold = null;   // {top, until}: where the frame's stage was before the answer
  function holdFrame() {
    frameHold = oneColumn.matches && stage && !stage.hidden
      ? { top: stage.getBoundingClientRect().top, until: Date.now() + 700 } : null;
  }
  function keepFrame() {
    if (!frameHold) return;
    if (Date.now() > frameHold.until) { frameHold = null; return; }
    var d = stage.getBoundingClientRect().top - frameHold.top;
    if (Math.abs(d) > 0.5) window.scrollBy(0, d);
  }
  function settleFrame() {
    if (!frameHold) return;
    keepFrame();
    requestAnimationFrame(function () { keepFrame(); requestAnimationFrame(keepFrame); });
    setTimeout(keepFrame, 150);
    setTimeout(keepFrame, 400);
  }
  function releaseFrame() { frameHold = null; }
  window.addEventListener('scroll', keepFrame, { passive: true });
  ['touchmove', 'wheel', 'keydown'].forEach(function (t) { window.addEventListener(t, releaseFrame, { passive: true, capture: true }); });

  // A decision button with its key; the frame keeps the focus (the bar is rebuilt) and its place.
  function decideButton(text, key, cls, onClick) {
    var b = button(text + ' ', function () { holdFrame(); onClick(); canvas.focus({ preventScroll: true }); settleFrame(); }, cls);
    if (key) b.appendChild(kbd(key));
    return b;
  }

  // One column while a box is selected and boxes still need a decision (or a check): the selected
  // box's main answers sit in the bar at the screen's foot, Save & next and its row go in the flow.
  // A dataset check suggestion on screen takes the bar with its two answers.
  function deciding() {
    if (!oneColumn.matches || !cur) return false;
    if (refineCur) return true;
    if (!boxes()[selected]) return false;
    var p = progress();
    return p.undecided + p.unchecked > 0;
  }

  // The decision bar: the selected box's question, its answers, the frame's progress.
  function renderDecision() {
    decide.textContent = '';
    var inBar = deciding();
    root.classList.toggle('rv-deciding', inBar);
    if (decideBar) { barBtns.textContent = ''; decideBar.hidden = !inBar; }
    placeAcceptAll();
    if (!cur) return;
    var sugBlock = suggestionBlock();
    if (sugBlock) decide.appendChild(sugBlock);
    // The dataset check's suggestion (none while an overlap suggestion shows): its answers take the
    // bar, the selected box's go in the card.
    var refBlock = refineBlock(inBar && !!decideBar);
    if (refBlock) decide.appendChild(refBlock);
    var p = progress();
    var b = boxes()[selected];
    if (inBar) renderBarHead(b, p);
    if (!b) {
      // A proposed box (an add) needs no "no box selected": its card says what to do.
      var off = dismissedOn ? 0 : dismissedCount();
      var tap = noHover.matches;
      if (!refBlock) decide.appendChild(el('div', 'rv-q', p.undecided || p.unchecked
        ? (tap ? 'No box selected: tap one, or turn on Draw to draw one.' : 'No box selected: j selects the next one, drag on the frame to draw one.')
        : (missingAsked === cur.frame ? missingText()
          : (p.total ? (tap ? 'Every box decided: Save & next saves.' : 'Every box decided: s saves.')
            : (tap ? 'No box: turn on Draw to draw one, or tick Confirmed empty below.' + (off ? ' ' + off + ' hidden.' : '')
              : 'No box: drag on the frame to draw one, or e confirms it empty.' + (off ? ' ' + off + ' hidden: i shows them.' : ''))))));
    } else {
      var i = selected;
      var q = el('div', 'rv-q');
      // Put-away boxes are not counted: box numbers keep their place, so the total would mislead.
      var putAway = boxes().some(isPutAway);
      q.appendChild(el('span', 'rv-q-num', 'Box ' + (i + 1) + (putAway ? '' : ' of ' + boxes().length) + ' · '));
      var verb, yes, no;
      if (b.source === 'human') { verb = 'Drawn: '; yes = 'Keep'; no = 'Delete'; }
      else if (b.state === 'accepted') { verb = 'Accepted as '; yes = 'Right, keep'; no = 'No, reject'; }
      else if (isDismissed(b)) { verb = (b.reason === 'spot_dismissed' ? 'Hidden at a spot you rejected: is it a ' : 'Hidden for its score: is it a '); yes = 'Yes, accept'; no = 'No, reject'; }
      else if (b.state === 'rejected') { verb = 'Rejected: is it a '; yes = 'Yes, accept'; no = 'Keep rejected'; }
      else { verb = 'Is this a '; yes = 'Yes, ' + b.label; no = 'No, reject'; }
      q.appendChild(document.createTextNode(verb));
      q.appendChild(el('strong', null, b.label));
      q.appendChild(document.createTextNode(verb.indexOf('Is') === 0 || verb.indexOf('is it') >= 0 ? '?' : ''));
      decide.appendChild(q);
      var why = [];
      if (b.auto_label && b.auto_label !== b.label) {
        why.push('auto-labelers say ' + b.auto_label + (b.auto_score != null ? ' (' + fmtScore(b.auto_score) + ')' : ''));
      }
      if (b.triage === 'review' && b.why && b.why.length) why.push(b.why.join('; '));
      else if (b.score != null) why.push('score ' + fmtScore(b.score));
      // Most verification errors are on small objects: say so (the close-up is at least 2x).
      if (Math.min(b.bbox[2] - b.bbox[0], b.bbox[3] - b.bbox[1]) < 24) why.unshift('small: check closely');
      if (b.difficult) why.unshift('marked unsure: kept out of training, its frame goes to held-out only, ignored by the benchmark');
      if (b.state === 'rejected' && b.reject_reason === 'too_small') why.unshift('too small: a real object, no negative; its frame goes to held-out only, ignored by the benchmark');
      if (b.state === 'rejected' && b.reject_reason) why.unshift('why: ' + reasonText(b.reject_reason));
      if (why.length) decide.appendChild(el('div', 'rv-q-why muted', why.join(' · ')));
      var rp = b.state === 'pending' ? repeatOf(b) : null;
      var btns = el('div', 'rv-q-btns');
      // The main answers: in the card, or (one column, deciding) in the bar at the foot, which
      // leaves the card the other answers as one compact row.
      var answers = inBar && decideBar && !refBlock ? barBtns : btns;
      answers.appendChild(decideButton(yes, 'a', 'ok' + (rp && rp.verdict === 'accepted' ? ' suggest' : ''), acceptSelected));
      answers.appendChild(decideButton(no, 'r', 'bad' + (rp && rp.verdict === 'rejected' ? ' suggest' : ''), rejectSelected));
      if (b.source !== 'human') {
        var parked = decideButton('Real, parked', 't', 'ok', parkSelected);
        parked.title = 'A real object that stays put: accept it and remember the spot, so that auto-label runs accept it there (same class, a close match) unless the VLM sees nothing';
        answers.appendChild(parked);
      }
      if (b.auto_label && b.auto_label !== b.label) {
        var al = b.auto_label;
        btns.appendChild(decideButton('It\'s a ' + al, null, 'ok', function () { relabelNext(i, al); }));
      }
      var other = decideButton('It\'s a…', '1-9', 'secondary', function () { classGrid = !classGrid; renderDecision(); });
      other.setAttribute('aria-expanded', classGrid ? 'true' : 'false');
      btns.appendChild(other);
      var unsure = decideButton(b.difficult ? 'Not unsure' : 'Unsure', 'd', 'secondary', toggleUnsure);
      unsure.title = 'A real object you cannot name for sure: kept out of training, its frame goes to held-out only, ignored by the benchmark';
      btns.appendChild(unsure);
      // A drawn box has nothing to reject: deleting it is the way out (Unsure still applies).
      if (b.source !== 'human') {
        var small = decideButton('Too small', null, 'secondary' + (b.reject_reason === 'too_small' ? ' active' : ''), tooSmall);
        small.title = 'A real object too small to label (same as reason ' + (reasonIndex('too_small') + 1) + ', Shift+' + (reasonIndex('too_small') + 1) +
          '): rejected but never a negative, its frame goes to held-out only, ignored by the benchmark';
        btns.appendChild(small);
      }
      decide.appendChild(btns);
      var mate = dupPartner(i);
      if (mate >= 0) {
        var mb = boxes()[mate];
        var dl = el('div', 'rv-q-dup');
        dl.appendChild(el('span', null, 'Overlaps box ' + (mate + 1) + ' (' + mb.label + (mb.score != null ? ' ' + fmtScore(mb.score) : '') + ')' +
          (dups.of[i] ? (dups.of[i].keeper === i ? ', this one kept as the best' : '') : ', not near-identical: keep both if they are two objects')));
        dl.appendChild(decideButton('Reject as duplicate', 'u', 'bad', function () { rejectWith(i, 'duplicate'); }));
        decide.appendChild(dl);
      }
      // Repeats: what the reviewer decided at this spot before, and the spot on every frame.
      if (b.source !== 'human') {
        var sl = el('div', 'rv-q-spot');
        var since = parkedSince(b);
        if (since) {
          sl.appendChild(el('span', null, 'Parked here since ' + since));
          var live = parkedSpot(b);
          if (live) {
            var forget = button('Forget this spot', function () { forgetSpot(live); }, 'secondary');
            forget.title = 'Auto-label runs stop accepting boxes here. The boxes already accepted stay.';
            sl.appendChild(forget);
          }
        }
        if (rp && rp.verdict) {
          sl.appendChild(el('span', null, (rp.verdict === 'rejected' ? 'Rejected' : 'Accepted') + ' here ' + rp.n +
            ' time(s) before' + (rp.reason ? ' (' + reasonText(rp.reason) + ')' : '') +
            (rp.total > rp.n ? ', ' + (rp.total - rp.n) + ' the other way' : '')));
        } else if (rp) {
          // Mixed: say so, suggest nothing (the same spot holds a real object at times).
          sl.appendChild(el('span', 'muted', 'Here before: rejected ' + rp.rej + ', accepted ' + rp.acc +
            (rp.reason ? ' (rejected as ' + reasonText(rp.reason) + ')' : '') + ': look closely'));
        }
        var everywhere = b.state === 'rejected' || (b.state === 'pending' && rp && rp.verdict === 'rejected') ? 'reject'
          : (b.state === 'accepted' || (b.state === 'pending' && rp && rp.verdict === 'accepted') ? 'accept' : null);
        // A duplicate says nothing about the spot: another box stays on the object there. Too
        // small neither: a real object, and a bulk reject would take frames out of training.
        if (b.state === 'rejected' && (b.reject_reason === 'duplicate' || b.reject_reason === 'too_small')) everywhere = null;
        // A dismissed box: nobody decided, so there is no decision to spread.
        if (isDismissed(b)) everywhere = null;
        if (everywhere) {
          var eb = button((everywhere === 'reject' ? 'Reject' : 'Accept') + ' this spot on all ' + camOf(cur.frame) + ' frames…',
            function () { spotEverywhere(i, everywhere === 'accept'); }, 'secondary');
          eb.title = 'Every unreviewed ' + camOf(cur.frame) + ' frame: the pending ' + b.label + ' boxes at this spot are ' +
            (everywhere === 'reject' ? 'rejected' : 'accepted (a closer match: IoU 0.8, the same class)') + '. Counts first and asks.';
          sl.appendChild(eb);
        }
        if (sl.childNodes.length) decide.appendChild(sl);
      }
      if (classGrid) {
        var g = el('div', 'rv-q-classes');
        visible().forEach(function (c, n) {
          var cb = button('', function () { classGrid = false; relabelNext(i, c); canvas.focus({ preventScroll: true }); }, 'secondary' + (c === b.label ? ' active' : ''));
          if (n < 10) cb.appendChild(kbd(String((n + 1) % 10)));
          cb.appendChild(document.createTextNode(' ' + c));
          g.appendChild(cb);
        });
        decide.appendChild(g);
      }
    }
    // Why the box just rejected (or the selected rejected box) was wrong: optional chips.
    var ri = reasonTarget();
    if (ri >= 0) {
      var rb = boxes()[ri];
      var rr = el('div', 'rv-q-reason');
      // One wrapping row of short chips (the full text is the tooltip): the box list keeps its room.
      var cap = el('span', 'muted', (ri === selected ? 'Why?' : (ri + 1) + ' why?'));
      cap.title = 'Optional: why box ' + (ri + 1) + ' is wrong' + (touchUi.matches ? '' : ' (Shift+1-9)');
      rr.appendChild(cap);
      REASONS.forEach(function (r, n) {
        var cb = button('', function () { setReason(n); canvas.focus({ preventScroll: true }); },
          'secondary' + (rb.reject_reason === r[0] ? ' active' : ''));
        cb.title = r[1] + (touchUi.matches ? '' : ' (Shift+' + (n + 1) + ')');
        cb.appendChild(kbd(String(n + 1)));
        cb.appendChild(document.createTextNode(r[2]));
        rr.appendChild(cb);
      });
      decide.appendChild(rr);
      // On a phone the chips are one row that scrolls: fade the edge with more chips past it.
      rr.setAttribute('data-scroll-cue', '');
      if (window.PrismLive && window.PrismLive.scrollCue) window.PrismLive.scrollCue(rr);
    }
    // Stacked boxes on the frame (in the progress line): keep the best of each stack, the rest
    // go as duplicates.
    var repeats = repeatRejects().length;
    var repeatBtn = null;
    if (repeats >= REPEAT_MIN) {
      repeatBtn = decideButton('Reject ' + repeats + ' repeats', null, 'secondary', rejectRepeats);
      repeatBtn.title = 'Reject the pending boxes of this frame that sit where you rejected boxes of this camera before';
    }
    var extra = dupDrops(dups).length;
    var stackBtn = null;
    if (extra) {
      stackBtn = decideButton('Reject ' + extra + ' stacked dup' + (extra === 1 ? '' : 's'), 'Shift+U', 'secondary', rejectDuplicates);
      stackBtn.title = 'Keep the best box of each of the ' + dups.groups.length + ' stack(s) of overlapping boxes, reject the others as duplicates';
    }
    // Overlapping boxes that are no stack: the reviewer's call, named so they can be found (the
    // boxes carry the `overlap` chip, the decision bar names the partner, `u` works on them).
    var lapping = (dups.overlaps || []).filter(function (o) { return boxes()[o[0]] && boxes()[o[1]]; });
    var lapNote = null;
    if (lapping.length) {
      lapNote = el('span', 'muted', 'Overlap: ' + lapping.slice(0, 3).map(function (o) { return (o[0] + 1) + '+' + (o[1] + 1); }).join(', ') +
        (lapping.length > 3 ? ' +' + (lapping.length - 3) + ' more' : ''));
      lapNote.title = 'These boxes of one object overlap or sit inside each other. Select one: ' +
        (touchUi.matches ? 'Reject as duplicate or No, reject under the question' : 'u rejects it as a duplicate, r rejects it as wrong') +
        '. Saving with both kept stops flagging the frame.';
    }
    // While the bar at the foot shows the progress, the card keeps only the stack / repeat tools.
    var showCount = p.total && !inBar;
    if (showCount || stackBtn || repeatBtn || lapNote) {
      var line = el('div', 'rv-progress');
      if (showCount) {
        var bar = el('progress');
        bar.max = p.total;
        bar.value = p.done;
        line.appendChild(bar);
      }
      var row = el('div', 'rv-progress-row');
      row.appendChild(el('span', 'muted', showCount ? p.done + '/' + p.total + ' done · ' + p.undecided + ' undecided' + (p.unchecked ? ' · ' + p.unchecked + ' to check' : '') : ''));
      if (lapNote) row.appendChild(lapNote);
      if (stackBtn) row.appendChild(stackBtn);
      if (repeatBtn) row.appendChild(repeatBtn);
      line.appendChild(row);
      decide.appendChild(line);
    }
  }

  // The bar's head: which box the answers are about and how far the frame is, pinned at the
  // screen's foot with them (the card with the full question may be under the fold).
  function renderBarHead(b, p) {
    barHead.textContent = '';
    var meter = el('span', 'rv-bar-meter');
    var fill = el('i');
    fill.style.width = (p.total ? Math.round(100 * p.done / p.total) : 0) + '%';
    meter.appendChild(fill);
    barHead.appendChild(meter);
    var q = el('span', 'rv-bar-q');
    var card = refineCur ? R.refineCard(refineCur.l, boxes(), true) : null;
    if (card) {
      // The answers under it are the suggestion's.
      q.textContent = card.short;
      barHead.appendChild(q);
      barHead.appendChild(el('span', 'rv-bar-count', refineCur.k + ' of ' + refineCur.n + ' suggestions'));
      barHead.title = p.done + ' of ' + p.total + ' boxes done';
      return;
    }
    if (b) {
      var putAway = boxes().some(isPutAway);
      q.appendChild(document.createTextNode('Box ' + (selected + 1) + (putAway ? '' : ' of ' + boxes().length) + ' · '));
      var state = b.source === 'human' ? 'drawn ' : (b.state === 'accepted' ? 'accepted ' : (b.state === 'rejected' ? 'rejected ' : ''));
      if (state) q.appendChild(document.createTextNode(state));
      q.appendChild(el('strong', null, b.label));
      if (!state) q.appendChild(document.createTextNode('?'));
    }
    barHead.appendChild(q);
    barHead.appendChild(el('span', 'rv-bar-count', p.undecided ? p.undecided + ' to decide'
      : (p.unchecked ? p.unchecked + ' to check' : 'all done')));
    barHead.title = p.done + ' of ' + p.total + ' boxes done';
  }

  function refresh() { renderSide(); draw(); }

  // Scrolls the box list so the selected box is in view; `page`: the page too, as little as needed
  // (not on a canvas click, which would move the frame under the pointer).
  function revealSelected(page) {
    var li = boxList.querySelector('li.sel');
    if (!li) return;
    var lr = li.getBoundingClientRect(), br = boxList.getBoundingClientRect();
    // An entry taller than the list shows from its top (its class select).
    if (lr.top < br.top || lr.height > br.height) boxList.scrollTop += lr.top - br.top - 4;
    else if (lr.bottom > br.bottom) boxList.scrollTop += lr.bottom - br.bottom + 4;
    // The page too when the decision card is off screen (one column: the panel does not stick).
    if (page) {
      var dr = decide.getBoundingClientRect();
      if (oneColumn.matches) {
        // The answers are in the bar at the screen's foot while boxes need a decision: the page
        // stays put, so the frame and the buttons are where they were for the next answer. Else
        // only a card scrolled away above (a tap in the box list far below) is brought back.
        if (!(decideBar && !decideBar.hidden) && dr.top < 0) { releaseFrame(); decide.scrollIntoView({ block: 'start' }); }
      } else if (dr.top < 0 || dr.bottom > window.innerHeight) decide.scrollIntoView({ block: 'nearest' });
    }
  }

  // Wide screens: the frame's CSS height is capped at the window less the line over it
  // (style.css reads `--rv-head`), so the line and the whole frame fit together.
  var metaRow = document.querySelector('.review-meta-row');
  function fitHead() {
    if (metaRow) canvas.style.setProperty('--rv-head', metaRow.offsetHeight + 'px');
  }

  // A new frame: the page scrolls as little as needed to show the whole frame and the side panel
  // (with its Save row; from the grid's top when they are taller than the window), the box list
  // starts at its top.
  function revealFrame() {
    releaseFrame();
    boxList.scrollTop = 0;
    fitHead();
    revealSelected();
    if (!grid) return;
    var top = grid.getBoundingClientRect().top;
    // One column: from the line over the frame (or the frame); the panel may go on below the screen.
    if (oneColumn.matches) {
      var head = phoneHead && phoneHead.getBoundingClientRect().height ? phoneHead : grid;
      top = head.getBoundingClientRect().top;
      // Every frame lands at the same spot, to the pixel: a frame shown 1 px lower than the last
      // is a frame that moved.
      if (Math.abs(top - 4) > 0.5) window.scrollBy(0, top - 4);
      return;
    }
    var bottom = Math.max(canvas.getBoundingClientRect().bottom, side ? side.getBoundingClientRect().bottom : 0);
    var by = 0;
    if (top < 0) by = top - 8;
    else if (bottom > window.innerHeight) by = Math.min(top - 8, bottom - window.innerHeight + 8);
    if (by) window.scrollBy(0, by);
  }

  // ---- edits --------------------------------------------------------------------------------

  function select(i, page) {
    var was = selected;
    selected = (i >= 0 && i < boxes().length) ? i : -1;
    // One column: the frame is the editor, so another box is zoomed to again even after a pan, a
    // pinch or an overview tap ended the follow (those hold the view for the box they were made
    // on; a page scroll started on the frame pans it). Fit stays the viewer's choice (pref).
    if (oneColumn.matches && selected !== was && boxes()[selected]) view.follow = pref('follow', true);
    // Touch: the reason chips stay until the selection moves on; Redraw is of the box it started on.
    if (reasonAsk && !reasonAsk.timer && selected !== reasonAsk.sel) clearReason(true);
    if (taps && taps.kind === 'redraw' && selected !== taps.idx) cancelTaps();
    // The selected box (or a proposed one: viewTarget).
    var vt = viewTarget();
    if (view.follow && vt) zoomTo(vt);
    else panTo(vt);
    refresh();
    revealSelected(page);
  }

  // Deciding a box (also confirming an earlier decision) checks it and moves the selection on to
  // the next box to review, then to the next decided box to double-check (if any).
  // `why`: the reviewer's reason for a rejection (markRejected).
  function setState(i, state, why) {
    var b = boxes()[i];
    if (!b) return;
    // Confirming a box already checked in that state only moves on (no undo step).
    if (!(b.state === state && b._checked)) {
      snapshot();
      if (state === 'rejected') markRejected(b, why);
      else { b.state = state; b._checked = true; delete b.reject_reason; endDismissal(b); }
      if (state !== 'rejected' && cur.labels.empty_confirmed) cur.labels.empty_confirmed = false;
    }
    var next = nextUndecided(i);
    select(next >= 0 ? next : i, true);
    sayProgress();
  }

  // Reject box `i` (`why`: the reason, see markRejected), then the next box; without a reason,
  // the decision bar offers the reason chips for a few seconds (never required: rejecting stays
  // one key).
  function rejectWith(i, why) {
    setState(i, 'rejected', why);
    var b = boxes()[i];
    if (!why && b && b.state === 'rejected' && b.source !== 'human') offerReason(i);
  }

  // Why a box was wrong (`reject_reason`, Shift+1-9), for hard-negative mining: the value, the
  // text, the chip's short text.
  var REASONS = [
    ['false_positive', 'nothing there', 'nothing'],
    ['insect_web', 'insect / web', 'insect'],
    ['rain_snow', 'rain / snow', 'rain'],
    ['ir_glare', 'IR glare / lights', 'glare'],
    ['reflection', 'reflection / picture', 'reflection'],
    ['static_object', 'part of the scene', 'scene'],
    ['other_object', 'other kind of object', 'other'],
    ['too_small', 'too small to tell', 'tiny'],
    ['duplicate', 'duplicate', 'dup']
  ];
  var REASON_MS = 4000;
  var reasonAsk = null;   // {frame, i, sel, timer}: the chips offered after a rejection

  function reasonIndex(r) {
    for (var n = 0; n < REASONS.length; n++) if (REASONS[n][0] === r) return n;
    return -1;
  }

  function reasonText(r) {
    for (var n = 0; n < REASONS.length; n++) if (REASONS[n][0] === r) return REASONS[n][1];
    return r;
  }

  function offerReason(i) {
    clearReason(true);
    // A finger needs longer than a key: on touch the chips stay until the selection changes or the
    // frame is saved (select, loadFrame), no timer.
    reasonAsk = {
      frame: cur.frame, i: i, sel: selected,
      timer: touchUi.matches ? null : setTimeout(function () { clearReason(); }, REASON_MS)
    };
    renderDecision();
  }

  function clearReason(quiet) {
    if (!reasonAsk) return;
    if (reasonAsk.timer) clearTimeout(reasonAsk.timer);
    reasonAsk = null;
    if (!quiet) renderDecision();
  }

  // The box the reason chips apply to: the one just rejected (while offered), else the selected
  // box when it is a rejected model / auto box; -1 for none.
  function reasonTarget() {
    if (!cur) return -1;
    var a = reasonAsk && reasonAsk.frame === cur.frame ? boxes()[reasonAsk.i] : null;
    if (a && a.state === 'rejected') return reasonAsk.i;
    var b = boxes()[selected];
    return b && b.state === 'rejected' && b.source !== 'human' ? selected : -1;
  }

  // Reason `n` of REASONS on the target box (again: cleared).
  function setReason(n) {
    var i = reasonTarget(), r = REASONS[n], b = boxes()[i];
    if (!b || !r) return;
    snapshot();
    endDismissal(b);
    if (b.reject_reason === r[0]) delete b.reject_reason; else b.reject_reason = r[0];
    say('Box ' + (i + 1) + ' rejected' + (b.reject_reason ? ': ' + r[1] : '') + '.');
    clearReason(true);
    refresh();
  }

  // The Too small button: the selected box is a real object too small to label. Rejected with
  // reason too_small, which makes it an ignore region (LabelBox::is_ignore), no hard negative. On
  // a box already rejected only the reason changes (again: cleared, like its reason chip), since
  // setState leaves a confirmed rejection as it is.
  function tooSmall() {
    var i = selected, b = boxes()[i];
    if (!b) { say('No box selected: j selects one.'); return; }
    if (b.state !== 'rejected') { rejectWith(i, 'too_small'); return; }
    snapshot();
    endDismissal(b);
    if (b.reject_reason === 'too_small') delete b.reject_reason; else b.reject_reason = 'too_small';
    say('Box ' + (i + 1) + ' rejected' + (b.reject_reason ? ': ' + reasonText('too_small') : '') + '.');
    clearReason(true);
    refresh();
  }

  // `d`: the selected box is a real object the reviewer cannot name for sure (VOC "difficult"):
  // an ignore region (LabelBox::is_ignore), so kept out of training, its frame held-out only,
  // ignored by the benchmark; then the next box. Again on an unsure box: no longer unsure (its
  // decision stays).
  function toggleUnsure() {
    var b = boxes()[selected];
    if (!b) { say('No box selected: j selects one.'); return; }
    snapshot();
    if (b.difficult) {
      delete b.difficult;
      refresh();
      say('Box ' + (selected + 1) + ' no longer marked unsure.');
      return;
    }
    b.difficult = true;
    delete b.reject_reason;
    endDismissal(b);
    b.state = b.source === 'human' ? 'added' : 'accepted';
    b._checked = true;
    if (cur.labels.empty_confirmed) cur.labels.empty_confirmed = false;
    var next = nextUndecided(selected);
    select(next >= 0 ? next : selected, true);
    sayProgress();
  }

  // The non-drawn boxes of the stacks that are not their keeper.
  function dupDrops(d) { return R.dupDrops(d, boxes()); }

  // Of those, the pending ones: what saving or Shift+A would otherwise accept as new duplicates.
  function pendingDupDrops(d) { return R.pendingDupDrops(d, boxes()); }

  // Shift+U: keep the best box of each stack, reject the others as duplicates (one undo step).
  function rejectDuplicates() {
    if (!cur) return;
    var d = dupGroups(), drop = dupDrops(d);
    if (!drop.length) { say('No stacked boxes to reject on this frame.'); return; }
    snapshot();
    drop.forEach(function (k) { markRejected(boxes()[k], 'duplicate'); });
    var sb = boxes()[selected];
    select(sb && sb.state !== 'rejected' ? selected : nextUndecided(selected), true);
    say('Rejected ' + drop.length + ' duplicate box(es) in ' + d.groups.length + ' stack(s); the best box of each stays for you to decide (z undoes).');
  }

  // `u`: reject the selected box as a duplicate of the box it overlaps.
  function rejectDupSelected() {
    if (!boxes()[selected]) { say('No box selected: j selects one; Shift+U clears every stack.'); return; }
    dups = dupGroups();
    if (dupPartner(selected) < 0) { say('Box ' + (selected + 1) + ' overlaps no other box of its object.'); return; }
    rejectWith(selected, 'duplicate');
  }

  // ---- repeats: the reviewer's earlier decisions at the same spot of this camera -------------

  // A spot counts as decided when at least REPEAT_MIN earlier decisions there (IoU >= REPEAT_IOU,
  // the same object class) agree REPEAT_SHARE of the time. Only a suggestion: nothing is decided
  // on load.
  var REPEAT_IOU = 0.6, REPEAT_MIN = 2, REPEAT_SHARE = 0.8;
  var ACCEPT_SPOT_IOU = 0.8;           // "accept this spot everywhere" (server: ACCEPT_SPOT_IOU)
  var histories = {};                  // camera -> {decisions} (GET /v1/review/history)

  function camOf(frame) { return String(frame || '').split('/')[1] || ''; }

  // The camera's history, once per camera and folder (kept up to date by noteHistory).
  function loadHistory(cam) {
    if (!cam || histories[cam]) return;
    histories[cam] = { decisions: [] };
    api('/v1/review/history?camera=' + encodeURIComponent(cam)).then(function (j) {
      histories[cam] = { decisions: j.decisions || [] };
      if (cur && camOf(cur.frame) === cam) refresh();
    }).catch(function () { /* no hint: the review works without it */ });
  }

  // A decision of the reviewer's own (the server's rule, repeats.rs `counts`). A duplicate is no
  // verdict on the spot: another box stays on the object there. Too small neither: a real object,
  // only too small to label.
  function lookedAt(b) {
    return b.source !== 'human' && (b.state === 'accepted' || b.state === 'rejected') &&
      b.reject_reason !== 'duplicate' && b.reject_reason !== 'too_small' &&
      !isDismissal(b.reason) &&
      ['low_score', 'other_class', 'repeat_spot', 'parked_spot'].indexOf(b.reason) < 0;
  }

  // After a reviewed save: the frame's decisions replace its earlier ones in the history.
  function noteHistory(frame, body, w, h) {
    var hist = histories[camOf(frame)];
    if (!hist) return;
    hist.decisions = hist.decisions.filter(function (d) { return d.frame !== frame; });
    body.boxes.filter(lookedAt).forEach(function (b) {
      hist.decisions.unshift({ frame: frame, label: b.label, bbox: b.bbox, state: b.state,
        reject_reason: b.reject_reason, width: w, height: h });
    });
  }

  // What the reviewer decided at box b's spot on the camera's other frames: null (fewer than
  // REPEAT_MIN decisions), else {verdict: 'accepted' | 'rejected' | null (mixed), n (of the
  // verdict), acc, rej, total, reason (the commonest reject reason)}.
  function repeatOf(b) {
    if (!cur || !b || b.source === 'human') return null;
    var hist = histories[camOf(cur.frame)];
    if (!hist) return null;
    var acc = 0, rej = 0, why = {};
    hist.decisions.forEach(function (d) {
      if (d.frame === cur.frame || d.width !== cur.width || d.height !== cur.height ||
        !sameObject(d.label, b.label) || iou(d.bbox, b.bbox) < REPEAT_IOU) return;
      if (d.state === 'accepted') acc++;
      else { rej++; if (d.reject_reason) why[d.reject_reason] = (why[d.reject_reason] || 0) + 1; }
    });
    var total = acc + rej;
    var top = Object.keys(why).sort(function (x, y) { return why[y] - why[x]; })[0] || null;
    if (total < REPEAT_MIN) return null;
    var r = { verdict: null, n: 0, acc: acc, rej: rej, total: total, reason: top };
    if (rej >= REPEAT_MIN && rej / total >= REPEAT_SHARE) { r.verdict = 'rejected'; r.n = rej; }
    else if (acc >= REPEAT_MIN && acc / total >= REPEAT_SHARE) { r.verdict = 'accepted'; r.n = acc; r.reason = null; }
    return r;
  }

  // ---- spots: what labelers runs remember of this camera (collect/labels/spots.rs) -----------

  var spots = {};                      // camera -> GET /v1/review/spots

  // The camera's spots, once per camera and folder.
  function loadSpots(cam) {
    if (!cam || spots[cam]) return;
    spots[cam] = { parked: {} };
    api('/v1/review/spots?camera=' + encodeURIComponent(cam)).then(function (j) {
      spots[cam] = j;
      if (cur && camOf(cur.frame) === cam) refresh();
    }).catch(function () { /* no line on the card: the review works without it */ });
  }

  function reloadSpots(cam) { delete spots[cam]; loadSpots(cam); }

  // The live parked spot that takes box b: the exact label at the accept IoU (spots::parked_at).
  function parkedSpot(b) {
    var s = cur && spots[camOf(cur.frame)];
    var list = s && s.parked ? s.parked[cur.width + 'x' + cur.height] || [] : [];
    return list.filter(function (p) {
      return !p.retired && p.label === b.label && iou(p.bbox, b.bbox) >= ACCEPT_SPOT_IOU;
    })[0] || null;
  }

  // The date of the parked spot behind box b: the run's mark, else a live spot it sits on.
  function parkedSince(b) {
    if (b.parked && b.parked.since) return b.parked.since;
    var p = parkedSpot(b);
    return p ? p.from : null;
  }

  // `t`: a real object that stays put: accept it (unsaved, like `a`) and remember the spot (at once,
  // so z undoes the accept only; the card's Forget this spot undoes the spot).
  function parkSelected() {
    var i = selected, b = boxes()[i];
    if (!cur || !b) { say('Select a box first: t marks the selected box as real and parked.'); return; }
    if (b.source === 'human') { say('A box you drew needs no spot: auto-label runs only look for their own boxes.'); return; }
    var cam = camOf(cur.frame), label = b.label;
    postJson('/v1/review/spots/parked', { frame: cur.frame, label: label, bbox: b.bbox }).then(function (j) {
      reloadSpots(cam);
      say(j.new ? 'Parked spot saved: auto-label runs accept ' + label + ' here from now on. ' : 'This spot was already marked parked, since ' + j.spot.from + '.');
      // Right here: `t` moved on to the next box, and z undoes the accept but not the spot.
      if (j.new) msg.appendChild(button('Forget this spot', function () { forgetSpot(j.spot); }, 'small secondary'));
    }).catch(function (e) { say('Accepted, but the spot was not saved: ' + e.message, true); });
    setState(i, 'accepted');
  }

  function forgetSpot(p) {
    if (!cur) return;
    var cam = camOf(cur.frame);
    postJson('/v1/review/spots/retire', { camera: cam, width: cur.width, height: cur.height, label: p.label, bbox: p.bbox }).then(function () {
      reloadSpots(cam);
      say('Spot forgotten: auto-label runs no longer accept boxes there.');
    }).catch(function (e) { say(e.message, true); });
  }

  // Panel options: the view changed, so what was learned about its places is wrong. Counts first.
  function cameraMoved() {
    var cam = cur ? camOf(cur.frame) : '';
    if (!cam) { say('Open a frame of the camera first.', true); return; }
    api('/v1/review/spots?camera=' + encodeURIComponent(cam)).then(function (j) {
      if (!window.confirm('Camera ' + cam + ' moved? This forgets ' + (j.live || 0) + ' spot(s): parked spots retire, and your rejections from before now stop hiding boxes. Reviewed frames stay as they are.')) return null;
      return postJson('/v1/review/spots/moved', { camera: cam }).then(function () {
        reloadSpots(cam);
        say('Spots of ' + cam + ' forgotten (' + (j.live || 0) + ').');
      });
    }).catch(function (e) { say(e.message, true); });
  }

  // The pending boxes of this frame the history says to reject.
  function repeatRejects() {
    var out = [];
    boxes().forEach(function (b, i) {
      if (b.state !== 'pending' || isUnwanted(b)) return;
      var r = repeatOf(b);
      if (r && r.verdict === 'rejected') out.push({ i: i, reason: r.reason });
    });
    return out;
  }

  // "Reject all repeats on this frame" (one undo step).
  function rejectRepeats() {
    var rs = repeatRejects();
    if (!rs.length) { say('No pending box here sits where you rejected boxes before.'); return; }
    snapshot();
    rs.forEach(function (r) { markRejected(boxes()[r.i], r.reason); });
    select(nextUndecided(-1), true);
    say('Rejected ' + rs.length + ' box(es) where you rejected boxes of this camera before (z undoes).');
  }

  // "Reject / accept this spot on every unreviewed frame of the camera": counts first, asks, then
  // the server edits the other frames and this frame's matching boxes change here (unsaved).
  function spotEverywhere(i, accept) {
    var b = boxes()[i];
    if (!cur || !b || b.source === 'human') return;
    var cam = camOf(cur.frame);
    var src = filtersForm.elements.source;
    var body = { camera: cam, label: b.label, bbox: b.bbox, width: cur.width, height: cur.height,
      skip_frame: cur.frame, dry_run: true };
    if (src && src.value) body.source = src.value;
    if (!accept && b.reject_reason) body.reject_reason = b.reject_reason;
    var url = accept ? '/v1/review/accept-spot' : '/v1/review/reject-spot';
    var verb = accept ? 'Accept' : 'Reject';
    say('Counting the frames...');
    postJson(url, body).then(function (j) {
      if (!j.boxes) {
        say('No other unreviewed ' + cam + ' frame has a pending ' + b.label + ' box at this spot.');
        return null;
      }
      if (!window.confirm(verb + ' ' + j.boxes + ' pending ' + b.label + ' box(es) at this spot on ' + j.eligible +
        ' unreviewed ' + cam + ' frame(s)?' + (accept ? '' : (b.reject_reason ? ' Reason: ' + reasonText(b.reject_reason) + '.' : '')) +
        ' The frames stay unreviewed; this frame changes here (save it as usual).')) {
        say('Nothing changed.');
        return null;
      }
      body.dry_run = false;
      return postJson(url, body).then(function (r) {
        // This frame: the same rule, here.
        var n = 0;
        snapshot();
        boxes().forEach(function (x, k) {
          var take = x.source !== 'human' && x.state === 'pending' &&
            (accept ? x.label === b.label && iou(x.bbox, b.bbox) >= ACCEPT_SPOT_IOU
              : sameObject(x.label, b.label) && iou(x.bbox, b.bbox) >= REPEAT_IOU);
          if (!take && k !== i) return;
          if (accept) { x.state = 'accepted'; x._checked = true; delete x.reject_reason; }
          else markRejected(x, b.reject_reason);
          if (k !== i) n++;
        });
        select(nextUndecided(-1), true);
        // Short: the panel's message line sits above the box list.
        say(verb + 'ed ' + (r.boxes || 0) + ' ' + b.label + ' box(es) here on ' + (r.changed || 0) + ' ' + cam + ' frame(s)' +
          (n ? ', ' + n + ' more on this one' : '') + ((r.failed || []).length ? '; ' + r.failed.length + ' not changed' : '') + '.');
        return loadNext(null, null, true);
      });
    }).catch(function (e) { say(e.message, true); });
  }

  // `a`: accept the selected box (a drawn one is kept), then the next one.
  function acceptSelected() {
    var b = boxes()[selected];
    if (!b) { say('No box selected: j selects one, Shift+A accepts every proposal.'); return; }
    setState(selected, b.source === 'human' ? 'added' : 'accepted');
  }

  // `r`: reject the selected box (a drawn one is deleted), then the next one.
  function rejectSelected() {
    var b = boxes()[selected];
    if (!b) return;
    if (b.source === 'human') { removeBox(selected); var n = nextUndecided(-1); if (n >= 0) select(n, true); sayProgress(); return; }
    rejectWith(selected, null);
  }

  function relabel(i, label) {
    var b = boxes()[i];
    if (!b || b.label === label) return;
    snapshot();
    if (b.source !== 'human' && !b.model_label) b.model_label = b.label;
    b.label = label;
    b._relabelled = true;
    b._checked = true;
    endDismissal(b);
    if (b.source !== 'human') b.state = 'accepted';
    if (cur.labels.empty_confirmed) cur.labels.empty_confirmed = false;
    refresh();
    sayProgress();
  }

  // "It's a ..." (the decision bar, 1-0): relabel the box (accepting it), then the next box.
  function relabelNext(i, label) {
    var b = boxes()[i];
    if (!b) return;
    relabel(i, label);
    var next = nextUndecided(i);
    select(next >= 0 ? next : i, true);
    sayProgress();
  }

  function removeBox(i) {
    if (!boxes()[i]) return;
    snapshot();
    boxes().splice(i, 1);
    selected = -1;
    refresh();
  }

  // `a`: accept the pending non-human boxes saving would accept anyway; hints stay pending, and a
  // box stacked on a better one is rejected as a duplicate instead (z undoes both).
  function acceptProposals() {
    if (!cur) return;
    var drop = pendingDupDrops(dupGroups());
    var todo = boxes().filter(function (b, i) {
      return drop.indexOf(i) < 0 && b.state === 'pending' && b.source !== 'human' && !isUnwanted(b) && !isReview(b) && resolved(b) === 'accepted';
    });
    var flagged = boxes().filter(function (b, i) { return drop.indexOf(i) < 0 && mustDecide(b); }).length;
    var left = flagged ? ' ' + flagged + ' flagged box(es) (marked ?) still need a decision.' : '';
    var hints = boxes().filter(function (b, i) { return drop.indexOf(i) < 0 && isHint(b) && !isUnwanted(b); }).length;
    if (!todo.length && !drop.length) {
      say((hints ? hints + ' hint box(es) stay pending (rejected on save unless you accept them); no other proposal.' : (flagged ? 'No proposal to accept.' : 'No pending proposal.')) + left);
      return;
    }
    snapshot();
    todo.forEach(function (b) { b.state = 'accepted'; });
    drop.forEach(function (k) { markDuplicate(boxes()[k]); });
    cur.labels.empty_confirmed = false;
    say('Accepted ' + todo.length + ' box(es)' +
      (drop.length ? '; rejected ' + drop.length + ' stacked duplicate(s) (z undoes)' : '') +
      (hints ? '; ' + hints + ' hint box(es) stay pending (rejected on save unless you accept them).' : '.') + left);
    refresh();
  }

  // `c`: accept every likely box of the frame; the others stay pending, and a likely box stacked on
  // a better one is left for the save to reject as a duplicate. They stay to check: a likely box is
  // not a sure one, 23% were false on public sets (z undoes).
  function acceptLikely() {
    if (!cur) return;
    var drop = pendingDupDrops(dupGroups());
    var todo = boxes().filter(function (b, i) { return drop.indexOf(i) < 0 && isLikely(b); });
    if (!todo.length) { say('No likely box on this frame.'); return; }
    snapshot();
    todo.forEach(function (b) { b.state = 'accepted'; });
    cur.labels.empty_confirmed = false;
    select(nextUndecided(-1), true);
    var u = undecidedCount();
    say('Accepted ' + todo.length + ' likely box(es)' + (u ? '; ' + u + ' still undecided' : '') + ' (z undoes).');
  }

  // `i`: show or put away this frame's dismissed boxes (view only: nothing changes until one is
  // decided).
  function toggleDismissed() {
    if (!cur) return;
    var n = dismissedCount();
    if (!n) { say('No hidden box on this frame.'); return; }
    dismissedOn = !dismissedOn;
    // Showing them clears the selection: the spotlight's veil would all but hide them.
    if (dismissedOn || (boxes()[selected] && isOff(boxes()[selected]))) selected = -1;
    if (boxes()[hoverIdx] && isOff(boxes()[hoverIdx])) hoverIdx = -1;
    refresh();
    say(dismissedOn ? n + ' hidden box(es) shown, dashed: click one, then a accepts it, r rejects it; i puts them away.' : 'Hidden boxes put away.');
  }

  // Shift+R: reject every box still undecided, for "the real ones are done, the rest are wrong".
  // Counts for the repeats history and spot memory like a rejection by hand (z undoes).
  function rejectRest() {
    if (!cur) return;
    var todo = boxes().filter(function (b) { return b.state === 'pending' && b.source !== 'human' && !isUnwanted(b); });
    if (!todo.length) { say('No undecided box.'); return; }
    snapshot();
    todo.forEach(function (b) { markRejected(b); b.reason = 'rest_rejected'; });
    say('Rejected ' + todo.length + ' undecided box(es) (z undoes).');
    refresh();
  }

  function confirmEmpty(on) {
    if (!cur) return;
    snapshot();
    if (on) {
      emptyBackup = { frame: cur.frame, boxes: JSON.stringify(boxes()) };
      cur.labels.empty_confirmed = true;
      cur.labels.boxes = boxes().filter(function (b) { return b.source !== 'human'; });
      boxes().forEach(function (b) { b.state = 'rejected'; b._checked = true; });
      selected = -1;
    } else {
      cur.labels.empty_confirmed = false;
      if (emptyBackup && emptyBackup.frame === cur.frame) {
        cur.labels.boxes = JSON.parse(emptyBackup.boxes);
        emptyBackup = null;
        if (selected >= boxes().length) selected = -1;
      }
    }
    refresh();
  }

  function doUndo() {
    if (!cur || !undo.length) return;
    // The selection goes back with the boxes: to the box the undone step changed.
    var u = JSON.parse(undo.pop());
    cur.labels = u.labels;
    lastNudge = null;
    // A suggestion answered in the undone step is back: the view frames it again.
    dups = dupGroups();
    suggestions = liveSuggestions();
    refineCur = currentRefine();
    select(u.selected, true);
    sayProgress();
  }

  function pickClass(n) {
    var c = visible()[n];
    if (c == null) return;
    if (selected >= 0) relabelNext(selected, c);
    else { classSel.value = c; say('New boxes: ' + c); }
  }

  // The first geometric edit of a model / auto box keeps its original box and accepts it.
  function markEdited(b) {
    b._checked = true;
    endDismissal(b);
    if (b.source !== 'human') {
      if (!b.orig_bbox) b.orig_bbox = b.bbox.slice();
      b.state = 'accepted';
    }
    cur.labels.empty_confirmed = false;
  }

  function moveBox(b, orig, dx, dy) {
    var W = canvas.width, H = canvas.height;
    var w = orig[2] - orig[0], h = orig[3] - orig[1];
    var x0 = clamp(orig[0] + dx, 0, Math.max(0, W - w)), y0 = clamp(orig[1] + dy, 0, Math.max(0, H - h));
    b.bbox = [x0, y0, x0 + w, y0 + h];
  }

  function resizeBox(b, orig, handle, dx, dy) {
    var W = canvas.width, H = canvas.height;
    var x0 = orig[0], y0 = orig[1], x1 = orig[2], y1 = orig[3];
    if (handle.indexOf('w') >= 0) x0 = clamp(Math.min(orig[0] + dx, x1 - MIN_BOX), 0, W);
    if (handle.indexOf('e') >= 0) x1 = clamp(Math.max(orig[2] + dx, x0 + MIN_BOX), 0, W);
    if (handle.indexOf('n') >= 0) y0 = clamp(Math.min(orig[1] + dy, y1 - MIN_BOX), 0, H);
    if (handle.indexOf('s') >= 0) y1 = clamp(Math.max(orig[3] + dy, y0 + MIN_BOX), 0, H);
    b.bbox = [x0, y0, x1, y1];
  }

  // Arrow keys: 1 px (Shift: 10 px) in frame pixels; a run of nudges is one undo step.
  function nudge(dx, dy) {
    var b = boxes()[selected];
    if (!b) return false;
    var now = Date.now();
    if (!(lastNudge && lastNudge.i === selected && now - lastNudge.t < 1000)) snapshot();
    else dirty = true;
    lastNudge = { i: selected, t: now };
    markEdited(b);
    moveBox(b, b.bbox.slice(), dx, dy);
    b.bbox = b.bbox.map(round1);
    refresh();
    return true;
  }

  // ---- canvas input -------------------------------------------------------------------------
  //
  // pointerdown starts a gesture:
  //   middle / right button, or Space + left: `pan` (drag moves the view);
  //   left on a handle of the selected box: `resize`; inside the selected box: `move`;
  //   left elsewhere: `draw`.
  // A gesture whose pointer moves less than CLICK_CSS is a click on pointerup (select the
  // smallest box under the point; a click on a handle keeps the selection). A move / resize
  // takes one undo snapshot when it starts moving.

  function canvasPoint(ev) {
    var r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) * canvas.width / (r.width || 1), (ev.clientY - r.top) * canvas.height / (r.height || 1)];
  }

  function framePoint(ev) {
    var c = canvasPoint(ev);
    var x = c[0] / view.z + view.ox, y = c[1] / view.z + view.oy;
    return [clamp(x, 0, canvas.width), clamp(y, 0, canvas.height)];
  }

  function handleAt(p) {
    var b = boxes()[selected];
    if (!b) return null;
    var css = HANDLE_CSS + 2;
    // A finger needs more room; the middle third of a box stays a move.
    if (touchUi.matches) css = Math.max(handleCss(b.bbox) + 2, Math.min(12, boxSideCss(b.bbox) / 3));
    var r = css * cssScale() / view.z;
    var hit = null;
    handlePoints(b.bbox).forEach(function (h) {
      if (!hit && Math.abs(p[0] - h[1]) <= r && Math.abs(p[1] - h[2]) <= r) hit = h[0];
    });
    return hit;
  }

  function insideSelected(p) {
    var b = boxes()[selected];
    return !!b && p[0] >= b.bbox[0] && p[0] <= b.bbox[2] && p[1] >= b.bbox[1] && p[1] <= b.bbox[3];
  }

  var HANDLE_CURSOR = { nw: 'nwse', se: 'nwse', ne: 'nesw', sw: 'nesw', n: 'ns', s: 'ns', e: 'ew', w: 'ew' };

  function setCursor(name) {
    if (canvas.getAttribute('data-cursor') !== name) canvas.setAttribute('data-cursor', name);
  }

  // The smallest shown box under the frame point `p`, else -1.
  function boxAt(p) {
    var best = -1, area = Infinity;
    if (hideBoxes) return -1;
    boxes().forEach(function (bx, i) {
      var a = (bx.bbox[2] - bx.bbox[0]) * (bx.bbox[3] - bx.bbox[1]);
      if (isShown(bx) && !isUnwanted(bx) && p[0] >= bx.bbox[0] && p[0] <= bx.bbox[2] && p[1] >= bx.bbox[1] && p[1] <= bx.bbox[3] && a < area) {
        best = i; area = a;
      }
    });
    return best;
  }

  // The shown box whose edge is nearest to the frame point `p`, within `reach` frame pixels, else -1.
  function boxNear(p, reach) {
    var best = -1, bestD = reach;
    if (hideBoxes) return -1;
    boxes().forEach(function (b, i) {
      if (!isShown(b)) return;
      var dx = Math.max(b.bbox[0] - p[0], 0, p[0] - b.bbox[2]), dy = Math.max(b.bbox[1] - p[1], 0, p[1] - b.bbox[3]);
      var d = Math.sqrt(dx * dx + dy * dy);
      if (d <= bestD) { best = i; bestD = d; }
    });
    return best;
  }

  // The hovered box shows its tag in full (also dimmed or checked ones).
  function setHover(i) {
    if (i === hoverIdx) return;
    hoverIdx = i;
    draw();
  }

  function hoverCursor(ev) {
    if (gesture) return;
    if (spaceDown) { setCursor('grab'); return; }
    if (!cur || hideBoxes || !ev) { setCursor('crosshair'); return; }
    var p = framePoint(ev);
    setHover(boxAt(p));
    var h = handleAt(p);
    if (h) setCursor(HANDLE_CURSOR[h]);
    else if (insideSelected(p)) setCursor('move');
    else setCursor('crosshair');
  }

  var lastPointer = null;
  var T = window.ReviewTouch;

  // A move / resize that already took its undo snapshot is undone exactly: the snapshot it pushed
  // goes back into the frame (box, `orig_bbox`, state, empty flag as before the gesture) and off
  // the undo list, and the frame is as dirty as it was. A draw is dropped; a pan keeps its view.
  function abortGesture() {
    var g = gesture;
    gesture = null;
    if (!g || !g.snapped || !undo.length) return;
    var u = JSON.parse(undo.pop());
    cur.labels = u.labels;
    selected = (u.selected >= 0 && u.selected < boxes().length) ? u.selected : -1;
    dirty = g.wasDirty;
    renderSide();
  }

  // Two fingers down: whatever the first one was doing stops, the view follows both from here.
  function startPinch(a, b) {
    abortGesture();
    gesture = {
      mode: 'pinch', a: a, b: b,
      start: { a: fingers[a], b: fingers[b], view: { z: view.z, ox: view.ox, oy: view.oy } }
    };
    view.follow = false;
    draw();
  }

  function applyPinch() {
    var g = gesture;
    var v = T.pinch(g.start, { a: fingers[g.a], b: fingers[g.b] }, { min: 1, max: MAX_ZOOM });
    view.z = v.z; view.ox = v.ox; view.oy = v.oy;
    clampView();
    draw();
  }

  // A pinch's finger lifted (or cancelled): the pinch ends, the fingers still down do nothing
  // until they lift, so the view never jumps to one finger's position.
  function fingerLifted() {
    gesture = Object.keys(fingers).length ? { mode: 'rest' } : null;
    draw();
  }

  // ---- Draw / Redraw by taps (touch UI; any pointer once turned on) ----

  var drawBtn = $('rv-draw');
  var redrawBtn = $('rv-redraw');
  var viewMenu = $('rv-view-menu');

  function cancelTaps(note) {
    if (!taps) return;
    var was = taps.kind;
    taps = null;
    if (note) say(was === 'redraw' ? 'Redraw cancelled.' : 'First corner cleared.');
    draw();
  }

  function setDrawMode(on) {
    drawMode = !!on;
    if (taps && taps.kind === 'draw') taps = null;
    say(drawMode ? 'Draw: drag, or tap two corners.' : 'Draw off: a finger selects, and pans a zoomed frame.');
    draw();
  }

  // Redraw: the next two taps are the selected box's corners.
  function startRedraw() {
    var b = boxes()[selected];
    if (!b) { say('No box selected: j selects one.'); return; }
    if (taps && taps.kind === 'redraw') { cancelTaps(true); return; }
    taps = { kind: 'redraw', idx: selected, first: null };
    say('Redraw box ' + (selected + 1) + ': tap one corner of the object, then the opposite one.');
    draw();
  }

  // A tap while Draw or Redraw waits for corners.
  function tapCorner(p) {
    if (!taps) taps = { kind: 'draw', first: null };
    if (!taps.first) {
      taps.first = p;
      say('First corner set: tap the opposite corner' + (taps.kind === 'redraw' ? ' of box ' + (taps.idx + 1) : '') + '.');
      draw();
      return;
    }
    var r = T.twoTapRect(taps.first, p, MIN_BOX);
    if (!r) {
      taps.first = p;
      say('Too close to the first corner: this tap is the first corner now; tap the opposite one.');
      draw();
      return;
    }
    var t = taps;
    taps = null;
    if (t.kind === 'redraw') {
      var b = boxes()[t.idx];
      if (!b) { draw(); return; }
      // As a drag resize: one undo step, a model / auto box keeps its original and is accepted.
      snapshot();
      markEdited(b);
      b.bbox = r.map(round1);
      lastNudge = null;
      selected = t.idx;
      refresh();
      say('Box ' + (t.idx + 1) + ' redrawn (z undoes).');
      return;
    }
    addDrawn(r);
    refresh();
    say('Box ' + (selected + 1) + ' drawn (z undoes). Tap two more corners for another.');
  }

  // A drawn box (a drag or two taps), selected.
  function addDrawn(r) {
    snapshot();
    boxes().push({
      label: classSel.value || classes[0] || 'object',
      bbox: r.map(round1),
      source: 'human', state: 'added', _checked: true
    });
    cur.labels.empty_confirmed = false;
    selected = boxes().length - 1;
  }

  canvas.addEventListener('pointerdown', function (ev) {
    if (!cur) return;
    canvas.focus({ preventScroll: true });
    var finger = ev.pointerType === 'touch';
    if (finger) {
      // No gesture: no finger is down (a lift the page missed is forgotten here).
      if (!gesture) fingers = {};
      fingers[ev.pointerId] = canvasPoint(ev);
      // A third finger is ignored; a second one starts a pinch (also after a pinch, from `rest`).
      if (gesture && gesture.mode === 'pinch') return;
      var ids = Object.keys(fingers);
      if (ids.length >= 2) {
        canvas.setPointerCapture(ev.pointerId);
        startPinch(Number(ids[0]), Number(ids[1]));
        return;
      }
    }
    if (ev.button === 1 || ev.button === 2 || (ev.button === 0 && spaceDown)) {
      ev.preventDefault();
      gesture = { mode: 'pan', id: ev.pointerId, cx: ev.clientX, cy: ev.clientY, ox: view.ox, oy: view.oy, moved: false };
      view.follow = false;
      canvas.setPointerCapture(ev.pointerId);
      setCursor('grabbing');
      return;
    }
    if (ev.button !== 0) return;
    var p = framePoint(ev);
    var g = { mode: 'draw', id: ev.pointerId, cx: ev.clientX, cy: ev.clientY, x0: p[0], y0: p[1], x1: p[0], y1: p[1], moved: false };
    g.finger = finger;
    var onHandle = null, inside = false;
    if (!hideBoxes && selected >= 0) {
      onHandle = handleAt(p);
      inside = !onHandle && insideSelected(p);
    }
    if (taps && taps.kind === 'redraw') {
      // Redraw waits for two taps; a drag in between pans a zoomed frame.
      g.tap = true;
      g.mode = view.z > 1 ? 'pan' : 'none';
    } else {
      // A mouse or a pen draws on empty frame as always; a finger only with Draw on
      // (ReviewTouch.touchIntent holds the rules).
      g.mode = T.touchIntent({ drawMode: drawMode || !finger, zoomed: view.z > 1, onHandle: !!onHandle, insideSelected: inside });
      g.tap = drawMode;
    }
    if (g.mode === 'resize') g.handle = onHandle;
    if (g.mode === 'move' || g.mode === 'resize') { g.idx = selected; g.orig = boxes()[selected].bbox.slice(); }
    if (g.mode === 'pan') { g.ox = view.ox; g.oy = view.oy; }
    gesture = g;
    canvas.setPointerCapture(ev.pointerId);
  });

  canvas.addEventListener('pointermove', function (ev) {
    lastPointer = ev;
    if (ev.pointerType === 'touch' && fingers[ev.pointerId]) fingers[ev.pointerId] = canvasPoint(ev);
    var g = gesture;
    if (!g) { hoverCursor(ev); return; }
    if (g.mode === 'pinch') {
      if (ev.pointerId === g.a || ev.pointerId === g.b) applyPinch();
      return;
    }
    if (g.mode === 'rest' || ev.pointerId !== g.id) return;
    if (!g.moved) {
      var still = g.finger ? TAP_CSS : CLICK_CSS;
      if (Math.abs(ev.clientX - g.cx) > still || Math.abs(ev.clientY - g.cy) > still) {
        g.moved = true;
        if ((g.mode === 'move' || g.mode === 'resize') && boxes()[g.idx]) {
          g.wasDirty = dirty;
          g.snapped = true;
          snapshot();
          markEdited(boxes()[g.idx]);
        }
      }
      if (!g.moved) return;
    }
    if (g.mode === 'none') return;
    if (g.mode === 'pan') {
      view.follow = false;
      var k = cssScale() / view.z;
      view.ox = g.ox - (ev.clientX - g.cx) * k;
      view.oy = g.oy - (ev.clientY - g.cy) * k;
      draw();
      return;
    }
    var p = framePoint(ev);
    g.x1 = p[0]; g.y1 = p[1];
    var b = boxes()[g.idx];
    if (g.mode === 'move' && b) moveBox(b, g.orig, g.x1 - g.x0, g.y1 - g.y0);
    else if (g.mode === 'resize' && b) resizeBox(b, g.orig, g.handle, g.x1 - g.x0, g.y1 - g.y0);
    draw();
  });

  canvas.addEventListener('pointerup', function (ev) {
    if (ev.pointerType === 'touch') delete fingers[ev.pointerId];
    var g = gesture;
    if (!g) return;
    if (g.mode === 'pinch' || g.mode === 'rest') {
      if (g.mode === 'rest' || ev.pointerId === g.a || ev.pointerId === g.b || !Object.keys(fingers).length) fingerLifted();
      return;
    }
    if (ev.pointerId !== g.id) return;
    gesture = null;
    // Draw / Redraw: a tap is a corner.
    if (g.tap && !g.moved) { tapCorner([g.x0, g.y0]); return; }
    // A finger's tap where it would pan still selects (below).
    if (g.mode === 'pan' && (g.moved || !g.finger)) { hoverCursor(ev); draw(); return; }
    if (g.mode === 'none' && g.moved) { draw(); return; }
    if (g.moved) {
      if (g.mode === 'draw') {
        var r = norm(g);
        if (r[2] - r[0] >= MIN_BOX && r[3] - r[1] >= MIN_BOX) {
          addDrawn(r);
          // A drag-drawn box replaces a first corner tapped before it.
          if (taps && taps.kind === 'draw') taps = null;
        }
      } else {
        var b = boxes()[g.idx];
        if (b) b.bbox = b.bbox.map(round1);
      }
      lastNudge = null;
      refresh();
      hoverCursor(ev);
      return;
    }
    if (g.mode === 'resize') { draw(); return; }
    // A click: the smallest box under the point (a finger: else the nearest box close by).
    var hit = boxAt([g.x0, g.y0]);
    if (hit < 0 && g.finger) hit = boxNear([g.x0, g.y0], TAP_NEAR_CSS * cssScale() / view.z);
    select(hit);
    hoverCursor(ev);
  });

  canvas.addEventListener('pointercancel', function (ev) {
    if (ev.pointerType === 'touch') delete fingers[ev.pointerId];
    var g = gesture;
    if (g && (g.mode === 'pinch' || g.mode === 'rest')) { fingerLifted(); return; }
    if (g && ev.pointerId !== g.id) return;
    gesture = null;
    // A cancelled move / resize keeps what it did (it is one undo step).
    if (g && g.moved && g.mode !== 'draw' && g.mode !== 'pan' && g.mode !== 'none') refresh(); else draw();
  });
  // A finger starting at the screen's left or right edge would swipe the browser back or forward
  // (the frame's `touch-action: none` does not stop that in Safari): those touches stay the page's.
  // Pointer events have already fired by then, so the gesture above still runs.
  canvas.addEventListener('touchstart', function (ev) {
    var t = ev.touches && ev.touches[ev.touches.length - 1];
    if (t && (t.clientX < 24 || t.clientX > window.innerWidth - 24)) ev.preventDefault();
  }, { passive: false });
  canvas.addEventListener('pointerenter', function () { overCanvas = true; });
  canvas.addEventListener('pointerleave', function () { overCanvas = false; if (!gesture) setHover(-1); });
  canvas.addEventListener('contextmenu', function (ev) { ev.preventDefault(); });
  canvas.addEventListener('mousedown', function (ev) { if (ev.button === 1) ev.preventDefault(); });
  canvas.addEventListener('auxclick', function (ev) { ev.preventDefault(); });
  canvas.addEventListener('wheel', function (ev) {
    if (!cur) return;
    var dy = ev.deltaY * (ev.deltaMode === 1 ? 16 : (ev.deltaMode === 2 ? 400 : 1));
    var factor = Math.exp(-dy * 0.0015);
    // Unzoomed, scrolling down scrolls the page.
    if (view.z <= 1 && factor <= 1) return;
    ev.preventDefault();
    view.follow = false;
    var c = canvasPoint(ev);
    zoomAt(c[0], c[1], factor);
  }, { passive: false });
  if (fitBtn) fitBtn.addEventListener('click', function () { fit(); canvas.focus({ preventScroll: true }); });
  if (zoomBoxBtn) {
    zoomBoxBtn.addEventListener('click', function () {
      if (!touchUi.matches) zoomToSelected();
      else if (zoomBoxFits()) {
        // One column remembers the choice for the next frames.
        if (view.follow && oneColumn.matches) setPref('follow', false);
        fit();
      } else {
        zoomToSelected();
        if (oneColumn.matches) setPref('follow', true);
      }
      canvas.focus({ preventScroll: true });
    });
  }
  if (drawBtn) drawBtn.addEventListener('click', function () { setDrawMode(!drawMode); });
  if (redrawBtn) redrawBtn.addEventListener('click', startRedraw);

  // ---- the View menu (touch UI): what the wheel and the view keys do ----

  // `l`, `v`, hold-`h`: one function each for the key and the menu.
  function toggleTags() { showTags = !showTags; draw(); }
  function toggleSpotlight() {
    spotlight = !spotlight;
    setPref('spotlight', spotlight);
    say(spotlight ? 'Spotlight on: the selected box stands out.' : 'Spotlight off: every box at full strength.');
    draw();
  }
  function zoomCentre(factor) {
    if (!cur) return;
    view.follow = false;
    zoomAt(canvas.width / 2, canvas.height / 2, factor);
  }
  var VIEW_ACTIONS = {
    'zoom-in': function () { zoomCentre(1.5); },
    'zoom-out': function () { zoomCentre(1 / 1.5); },
    fit: function () { fit(); },
    image: cycleImg,
    hide: function () { hideBoxes = !hideBoxes; draw(); },
    labels: toggleTags,
    spotlight: toggleSpotlight,
    brighter: function () { stepGamma(1); },
    darker: function () { stepGamma(-1); },
    reset: function () { setImgView('auto', 1); }
  };
  // Fit whole frame and the image setting (`b`): an older page gets them made here.
  var viewItems = viewMenu && viewMenu.querySelector('.rv-view-items');
  if (viewItems) {
    [['fit', 'Fit whole frame', 'hide'], ['image', 'Image', 'brighter']].forEach(function (v) {
      if (viewItems.querySelector('button[data-view="' + v[0] + '"]')) return;
      var vb = el('button', 'small secondary', v[1]);
      vb.type = 'button';
      vb.setAttribute('data-view', v[0]);
      viewItems.insertBefore(vb, viewItems.querySelector('button[data-view="' + v[2] + '"]'));
    });
  }
  if (viewMenu) {
    viewMenu.addEventListener('click', function (ev) {
      var btn = ev.target.closest && ev.target.closest('button[data-view]');
      var act = btn && VIEW_ACTIONS[btn.getAttribute('data-view')];
      if (act) act();
    });
    // Open, it closes on a tap anywhere else, and on Esc.
    document.addEventListener('pointerdown', function (ev) {
      if (viewMenu.open && !viewMenu.contains(ev.target)) viewMenu.open = false;
    });
    viewMenu.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' && viewMenu.open) { viewMenu.open = false; viewMenu.querySelector('summary').focus(); ev.stopPropagation(); }
    });
  }

  // The touch controls show on the touch UI only, with their state; the wording that names keys,
  // the wheel or a hover follows it too.
  var canvasLabel = canvas.getAttribute('aria-label') || '';
  var legendHover = $('rv-legend-hover');
  var legendText = legendHover ? legendHover.textContent : '';
  function touchViewUi() {
    var on = touchUi.matches;
    if (drawBtn) {
      drawBtn.hidden = !on;
      drawBtn.setAttribute('aria-pressed', drawMode ? 'true' : 'false');
    }
    if (redrawBtn) {
      redrawBtn.hidden = !on || !boxes()[selected];
      redrawBtn.setAttribute('aria-pressed', taps && taps.kind === 'redraw' ? 'true' : 'false');
    }
    if (viewMenu) {
      viewMenu.hidden = !on;
      var pressed = { hide: hideBoxes, labels: showTags, spotlight: spotlight };
      Array.prototype.forEach.call(viewMenu.querySelectorAll('button[data-view]'), function (b) {
        var k = b.getAttribute('data-view');
        if (k in pressed) b.setAttribute('aria-pressed', pressed[k] ? 'true' : 'false');
        if (k === 'image') b.textContent = imgLabel();
      });
    }
  }
  function touchWords() {
    var on = touchUi.matches;
    canvas.setAttribute('aria-label', on
      ? 'The frame with its boxes. Tap a box to select it, drag its handles to resize it, drag inside it to move it; pinch to zoom, drag a zoomed frame to pan; turn on Draw to draw a new box (drag, or tap two corners).'
      : canvasLabel);
    if (legendHover) legendHover.textContent = on ? 'a selected box shows its full tag' : legendText;
  }
  function touchUiChanged() {
    // Off the touch UI the toggles have no button: they go off.
    if (!touchUi.matches) { drawMode = false; taps = null; if (viewMenu) viewMenu.open = false; }
    touchWords();
    placeAcceptAll();
    if (cur) refresh(); else touchViewUi();
  }
  [touchUi, oneColumn].forEach(function (m) {
    if (m.addEventListener) m.addEventListener('change', touchUiChanged);
    else if (m.addListener) m.addListener(touchUiChanged);
  });
  touchWords();
  touchViewUi();

  // "Accept proposals" (Shift+A): on one column the bar shows "Accept likely" in its place when
  // the frame has likely boxes, and it moves to the row of Skip / Back / Next (not while deciding:
  // then the answers have the bar and Accept likely is beside the box list).
  var acceptAllBtn = $('rv-accept-all');
  var moreRow = document.querySelector('.rv-actions-more');
  function placeAcceptAll() {
    if (!acceptAllBtn || !likelyBar || !moreRow) return;
    // While deciding, Accept likely is beside the box list (the bar holds the answers): Accept
    // proposals keeps its place in the row of Save & next.
    var away = oneColumn.matches && !likelyBar.hidden && !root.classList.contains('rv-deciding');
    if (away && acceptAllBtn.parentNode !== moreRow) moreRow.insertBefore(acceptAllBtn, moreRow.firstChild);
    else if (!away && acceptAllBtn.parentNode === moreRow) likelyBar.parentNode.insertBefore(acceptAllBtn, likelyBar.nextSibling);
  }

  // "Larger close-up": a wider side panel (per viewer).
  var wideBtn = $('rv-crop-wide');
  if (!wideBtn && cropWrap) {
    wideBtn = button('', function () {}, 'secondary rv-crop-wide');
    wideBtn.id = 'rv-crop-wide';
    cropWrap.appendChild(wideBtn);
  }
  function applyWide() {
    if (grid) grid.classList.toggle('rv-wide', wideSide);
    if (wideBtn) {
      wideBtn.textContent = wideSide ? 'Smaller close-up' : 'Larger close-up';
      wideBtn.setAttribute('aria-pressed', wideSide ? 'true' : 'false');
    }
  }
  if (wideBtn) {
    wideBtn.addEventListener('click', function () {
      wideSide = !wideSide;
      setPref('wide', wideSide);
      applyWide();
      draw();
      canvas.focus({ preventScroll: true });
    });
  }
  applyWide();
  // The close-up follows the side panel's width.
  var resizeTimer = null;
  window.addEventListener('resize', function () {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () { resizeTimer = null; fitHead(); draw(); }, 100);
  });

  // ---- loading and saving -------------------------------------------------------------------

  function showCounts(j) {
    var c = j.counts || {};
    var box = $('rv-counts');
    box.textContent = '';
    var add = function (k, v) {
      var s = el('span');
      s.appendChild(el('span', 'k', k + ' '));
      s.appendChild(document.createTextNode(String(v)));
      box.appendChild(s);
      return s;
    };
    // A big folder's index still filling in: the counts and the order are of the frames so far.
    if (j.indexing) add('Indexing', indexingRest(j.indexing)).id = 'rv-indexing';
    add('Reviewed', c.reviewed || 0);
    add('Remaining', c.remaining || 0);
    add('Skipped', c.skipped || 0);
    add('Confirmed empty', c.empty_confirmed || 0);
    add('Rejected boxes', c.rejected_boxes || 0);
    if (c.auto_review != null) add('Auto: to review', c.auto_review);
    if (c.auto_accept != null) add('Auto: accept-ready', c.auto_accept);
    if (c.repeats_only) add('Repeat-only', c.repeats_only);
    // The boxes kept back from dismissal: how often the filter would have hidden a real object.
    // A percentage of a few boxes says nothing, so none under 20.
    var real = c.audit_accepted || 0, audited = real + (c.audit_rejected || 0);
    if (audited) {
      add('Hidden-box check:', real + ' of ' + audited + (real === 1 ? ' was' : ' were') + ' real' +
        (audited >= 20 ? ' (' + Math.round(real / audited * 100) + '%)' : ''));
    }
    var per = c.per_class || {};
    var keys = Object.keys(per);
    add('Per class', keys.length ? keys.map(function (k) { return k + ' ' + per[k]; }).join(', ') : 'none yet');
    add('Matching', j.position ? (j.position + ' of ' + j.total) : (j.total || 0));
    if (j.unreadable) add('Unreadable', j.unreadable);
    queuePlace = j.position ? (j.position + ' of ' + j.total) : '';
    // The counts are behind Details on a phone: the line says the order is still filling in.
    phoneIndexing = j.indexing ? 'indexing' : '';
    setPhoneLine();
  }

  // Phones: one line over the frame; Details opens the counts, the filters and the frame's facts.
  var queuePlace = '', phoneIndexing = '';
  function setPhoneLine() {
    if (!phoneLine) return;
    var r = (cur && cur.record) || {};
    phoneLine.textContent = [queuePlace ? 'Frame ' + queuePlace : '', phoneIndexing, r.scene, r.camera].filter(Boolean).join(' · ') || 'Review';
    phoneLine.title = $('rv-meta').textContent;
  }

  function fillCameras(list) {
    var sel = $('rv-camera');
    var keep = sel.value;
    var have = Array.prototype.map.call(sel.options, function (o) { return o.value; });
    (list || []).forEach(function (c) {
      if (have.indexOf(c) < 0) { var o = el('option', null, c); o.value = c; sel.appendChild(o); }
    });
    sel.value = keep;
  }

  // The class pickers' chips: the catalog's entries, the class list and the classes the queue
  // saw (model labels outside the list too).
  function buildEntries() {
    var base = catalog && catalog.entries ? catalog.entries : [];
    if (!CU) return [];
    var out = CU.mergeEntries(base, classes.concat(seenLabels));
    out.forEach(function (e) {
      if (classes.indexOf(e.name) >= 0) e.in_list = true;
      e.hidden = hiddenClasses.indexOf(e.name) >= 0;
    });
    return out;
  }

  // The Class filter: remember the classes the queue saw. Drawn again only when its entries
  // change: the counts refresh while a folder is indexed comes every 2 s, maybe with it open.
  var labelsDrawn = '';
  function fillLabels(list) {
    (list || []).forEach(function (c) { if (seenLabels.indexOf(c) < 0) seenLabels.push(c); });
    if (!labelPicker) return;
    var entries = buildEntries(), key = JSON.stringify(entries);
    if (key === labelsDrawn) return;
    labelsDrawn = key;
    labelPicker.setEntries(entries);
  }

  function fillClassSelect() {
    var keep = classSel.value;
    classSel.textContent = '';
    visible().forEach(function (c, i) {
      var o = el('option', null, (i + 1) + '. ' + c);
      o.value = c;
      classSel.appendChild(o);
    });
    if (visible().indexOf(keep) >= 0) classSel.value = keep;
  }

  function fillKeyHint() {
    var shown = visible().slice(0, 10).map(function (c, i) { return ((i + 1) % 10) + ' ' + c; });
    Array.prototype.forEach.call(document.querySelectorAll('.rv-key-classes, #rv-key-classes'), function (k) {
      k.textContent = shown.length ? ' (' + shown.join(', ') + ')' : '';
    });
  }

  function setMeta() {
    var m = $('rv-meta');
    if (!cur) { m.textContent = ''; return; }
    var r = cur.record || {};
    var p = r.provenance;
    var origin = p
      ? 'imported (' + (p.url || p.dataset || p.path || r.source) + (p.time_offset_s != null ? ' at ' + p.time_offset_s + ' s' : '') + '; licence ' + (p.license || '?') + ')'
      : r.model + ' on ' + r.device;
    var parts = [cur.frame, r.timestamp, r.scene, r.camera, origin,
      p ? '' : 'threshold ' + fmtScore(r.threshold), (r.buckets || []).join(', '),
      cur.saved ? ('saved: ' + cur.labels.status) : 'not reviewed yet'];
    var ar = cur.labels && cur.labels.auto_run;
    if (ar) parts.push('auto-labelled' + (ar.labeler ? ' by ' + ar.labeler : '') + (ar.at ? ' ' + ar.at : ''));
    if (ar && ar.classes && ar.classes.length && !cur.labels.boxes.some(function (b) { return b.source === 'auto'; })) {
      parts.push('auto-label found nothing of: ' + ar.classes.join(', '));
    }
    if (cur.labels && cur.labels.bulk_accepted) parts.push('bulk accepted');
    if (cur.stale_classes) parts.push('reviewed with another class list');
    if (cur.labels && cur.labels.disagreement) parts.push('back for review: ' + cur.labels.disagreement);
    m.textContent = parts.filter(Boolean).join(' · ');
    setPhoneLine();
  }

  function showNone(on) {
    $('rv-none').hidden = !on || !!missingNote.textContent;
    missingNote.hidden = !on || !missingNote.textContent;
    canvas.hidden = on;
    if (on && cropWrap) cropWrap.hidden = true;
  }

  // The folder under review is gone (an unplugged drive): that, not "no frame matches".
  var missingNote = el('p', 'form-msg error-text');
  missingNote.hidden = true;
  $('rv-none').parentNode.insertBefore(missingNote, $('rv-none'));
  function noteMissing(text) {
    missingNote.textContent = text || '';
    if (!$('rv-none').hidden || !missingNote.hidden) showNone(true);
  }

  // A big folder's index filling in (`indexing` of GET /v1/review/next, absent once done): its
  // progress in the counts, and in place of "no frame matches" while no frame has come in yet (in
  // the same paragraph, so it shows where and as that does). A counts refresh every 2 s follows it
  // until it is done, the page is hidden or left.
  var noneText = el('span'), indexNote = el('span');
  while ($('rv-none').firstChild) noneText.appendChild($('rv-none').firstChild);
  indexNote.id = 'rv-indexing-none';
  indexNote.hidden = true;
  $('rv-none').appendChild(noneText);
  $('rv-none').appendChild(indexNote);
  var indexing = null, indexTimer = null;
  var INDEX_POLL_MS = 2000;
  // The frames are still being counted at first: no "0 of 0".
  function indexingRest(ix) {
    var of = ix.total ? (ix.done || 0) + ' of ' + ix.total + ' frames' : 'the frames';
    return of + '. Counts and order are still filling in.';
  }
  function stopIndexPoll() {
    if (indexTimer) clearTimeout(indexTimer);
    indexTimer = null;
  }
  function followIndexing(ix) {
    indexing = ix || null;
    indexNote.textContent = indexing ? 'Indexing ' + indexingRest(indexing) : '';
    indexNote.hidden = !indexing;
    noneText.hidden = !!indexing;
    pollIndexing();
  }
  // A request a timer makes, not the reviewer (the poll header of live.js).
  function background(fn) {
    return window.PrismPoll && window.PrismPoll.background ? window.PrismPoll.background(fn) : fn();
  }
  function pollIndexing() {
    stopIndexPoll();
    if (!indexing || document.hidden) return;
    indexTimer = setTimeout(function () {
      indexTimer = null;
      // A poll: it does not keep the session alive (`X-Prism-Poll`, live.js).
      background(function () { loadNext(null, null, true); });
    }, INDEX_POLL_MS);
  }
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) stopIndexPoll();
    else if (indexing && !indexTimer) background(function () { loadNext(null, null, true); });
  });
  // Leaving the tab (or the page kept in the back-forward cache): no refresh in the background.
  window.addEventListener('pagehide', stopIndexPoll);
  window.addEventListener('pageshow', function (e) { if (e.persisted) pollIndexing(); });

  function prefetch() {
    var next = upcoming[0];
    if (!next || (cur && next === cur.frame)) return;
    prefetchImg = new Image();
    prefetchImg.src = '/dataset/image?path=' + encodeURIComponent(next);
  }

  // Frames on their way to the screen (`loadFrame`): a counts refresh does not land over one.
  var framesLoading = 0;

  // `nav`: 'back' (`p`) or 'forward' (`n` after Back); else a move on, which forgets the frames
  // Back left.
  function loadFrame(frame, nav) {
    clearReason(true);
    framesLoading++;
    var settled = function () { framesLoading--; };
    return api('/v1/review/labels?frame=' + encodeURIComponent(frame)).then(function (j) {
      if (cur && cur.frame !== j.frame) {
        if (nav === 'back') fwdStack.push(cur.frame);
        else { backStack.push(cur.frame); if (nav !== 'forward') fwdStack = []; }
        if (backStack.length > 200) backStack.shift();
      }
      cur = j;
      // Suggestions keep to the boxes the server named (`overlap_live_boxes`).
      R.markServed(j.labels);
      loadHistory(camOf(j.frame));
      loadSpots(camOf(j.frame));
      if (Array.isArray(j.classes)) classes = j.classes;
      if (Array.isArray(j.hidden)) hiddenClasses = j.hidden;
      undo = []; dirty = false; guard = null; lastNudge = null; gesture = null; taps = null;
      emptyBackup = null; missingAsked = null; frameStart = Date.now();
      // Asking for these frames means asking to see those boxes.
      dismissedOn = !!filtersForm.elements.auto && filtersForm.elements.auto.value === 'dismissed';
      canvas.width = j.width || 640;
      canvas.height = j.height || 360;
      // The CSS size keeps this ratio (style.css caps a tall frame at the window's height).
      canvas.style.setProperty('--rv-ar', String(canvas.width / canvas.height));
      overview.style.setProperty('--rv-ar', String(canvas.width / canvas.height));
      view.z = 1; view.ox = 0; view.oy = 0;
      hoverIdx = -1;
      selected = nextUndecided(-1);
      // A suggestion opens on the box it would drop.
      dups = dupGroups();
      suggestions = liveSuggestions();
      if (suggestions.length) selected = suggestionBox(suggestions[0]);
      // Else a dataset check suggestion opens on its box (none for an added box: the proposal).
      refinePutOff = [];
      refineCur = currentRefine();
      if (refineCur) selected = R.refineFocus(refineCur.l, boxes()).mark;
      var draftNote = restoreDraft(j);
      // One column: following is the viewer's choice again (a pinch on the last frame stopped it).
      if (oneColumn.matches) view.follow = pref('follow', true);
      // Following (`x`): the new frame opens zoomed on its first box (or the suggestion's place).
      var opening = viewTarget();
      if (view.follow && opening) zoomTo(opening);
      img = new Image();
      img.onload = draw;
      img.src = j.image_url;
      showNone(false);
      setMeta();
      refresh();
      revealFrame();
      prefetch();
      if (draftNote) say(draftNote[0], draftNote[1]);
    }).then(settled, function (e) { settled(); throw e; });
  }

  // ---- drafts: unsaved edits across a discarded tab -------------------------------------------
  // A phone discards a background tab and reloads it later; `beforeunload` does not run there. So
  // when the page goes out of view (`pagehide`, hidden) the frame's unsaved edits are kept in
  // sessionStorage (PrismDraft, live.js) with the frame's revision. The next load opens that frame
  // and restores them only at the same revision (else nobody's edits would be overwritten
  // unseen: they are dropped and the page says so). The labels come back as the undo stack holds
  // them (the `_` page fields such as `_n` and `_checked` too), so they save exactly as if edited
  // live; one undo step returns to the frame as served. A save or another frame drops the draft.
  var DRAFT = 'review';
  var draftWaiting = false;   // the draft's frame is loading: nothing is written or dropped meanwhile
  function keepDraft() {
    if (!window.PrismDraft || draftWaiting) return;
    if (cur && dirty) PrismDraft.save(DRAFT, { id: cur.frame, rev: cur.revision, labels: cur.labels, selected: selected });
    else PrismDraft.drop(DRAFT);
  }
  // In loadFrame, once `cur` is the served frame: [message, isError] when a draft was restored or
  // dropped, else null.
  function restoreDraft(j) {
    if (!window.PrismDraft) return null;
    // Only the page's first frame takes a draft back; any later load is a frame change (a
    // reload of this frame included: its edits were let go).
    if (!draftWaiting) { PrismDraft.drop(DRAFT); return null; }
    var d = PrismDraft.load(DRAFT), v = PrismDraft.verdict(d, j.frame, j.revision);
    if (v === 'none') return null;
    PrismDraft.drop(DRAFT);
    if (v === 'other') return null;
    if (v === 'drop' || !d.labels || !Array.isArray(d.labels.boxes)) {
      return ['Your unsaved edits on this frame were dropped: someone saved it since.', true];
    }
    snapshot();   // one undo step back to the frame as served; marks the frame unsaved
    cur.labels = d.labels;
    selected = (typeof d.selected === 'number' && d.selected >= 0 && d.selected < boxes().length) ? d.selected : -1;
    dups = dupGroups();
    suggestions = liveSuggestions();
    refineCur = currentRefine();
    return ['Restored your unsaved edits from before the page was closed.', false];
  }
  window.addEventListener('pagehide', keepDraft);
  document.addEventListener('visibilitychange', function () { if (document.hidden) keepDraft(); });

  // The walk of /v1/review/next running ({countsOnly, stale: answer ignored, noLanding: counts
  // only, done: promise}), and the newest intent waiting for it (R.nextPlan decides).
  var nextRun = null, nextWaiting = null;

  // A real Next is on its way: another press would only repeat the same walk.
  function nextComing() {
    return !!((nextRun && !nextRun.countsOnly) || (nextWaiting && !nextWaiting.countsOnly));
  }

  // The next frame after `after` (the first without), `note` said first; `countsOnly` refreshes
  // the counts and lists and keeps the frame on screen (and its edits). At most one walk at a time
  // for a real Next; the promise ends once what this call asked for, or what replaced it, is done.
  function loadNext(after, note, countsOnly) {
    var want = { after: after, note: note, countsOnly: !!countsOnly };
    var kind = function (w) { return w ? (w.countsOnly ? 'counts' : 'real') : null; };
    var plan = R.nextPlan(kind(nextRun), kind(nextWaiting), kind(want));
    if (plan === 'drop') return nextRun.done;
    if (plan === 'queue') {
      if (!want.countsOnly) nextRun.noLanding = true;
      nextWaiting = want;
      return nextRun.done;
    }
    if (plan === 'jump') { nextRun.stale = true; nextWaiting = null; }
    return runNext(want);
  }

  function runNext(want) {
    var run = { countsOnly: want.countsOnly, stale: false, noLanding: false, done: null };
    var after = want.after, note = want.note, countsOnly = want.countsOnly;
    nextRun = run;
    var p = filterParams();
    if (after) p.set('after', after);
    run.done = api('/v1/review/next?' + p.toString()).then(function (j) {
      if (run.stale) return;
      showCounts(j);
      fillCameras(j.cameras);
      fillLabels(j.labels);
      noteMissing(j.folder_missing ? j.message : '');
      followIndexing(j.indexing);
      var shown = !!cur || framesLoading > 0;
      if (R.nextLanding(countsOnly ? 'counts' : 'real', run.noLanding, shown) === 'counts') return;
      upcoming = Array.isArray(j.upcoming) ? j.upcoming : [];
      var text = note || '';
      if (j.wrapped) text = (text ? text + ' ' : '') + 'Back to the first frame of the queue.';
      if (text || note === '') say(text);
      if (!j.item) {
        if (cur) backStack.push(cur.frame);
        cur = null; img = null; undo = []; dirty = false; selected = -1;
        showNone(true); setMeta(); refresh();
        return;
      }
      return loadFrame(j.item.frame);
    }).catch(function (e) {
      if (run.stale) return;
      say(e.message, true);
      // A failed refresh does not end the indexing message: the next one may answer.
      if (indexing) pollIndexing();
    }).then(function () {
      if (nextRun !== run) return;
      var w = nextWaiting;
      nextRun = null;
      nextWaiting = null;
      if (w) return runNext(w);
    });
    return run.done;
  }

  // After a job or a bulk accept: a frame with unsaved edits stays, else the queue reloads.
  function reloadQueue() {
    if (cur && dirty) return loadNext(null, null, true);
    return loadNext();
  }

  function showConflict(text) {
    msg.textContent = '';
    msg.className = 'form-msg error-text';
    msg.appendChild(document.createTextNode(text + ' '));
    var frame = cur && cur.frame;
    msg.appendChild(button('Reload frame', function () {
      if (!frame) return;
      loadFrame(frame).then(function () { say('Reloaded the frame; your edits were dropped.'); })
        .catch(function (e) { say(e.message, true); });
    }));
  }

  function save(status) {
    if (!cur || busy) return;
    if (status === 'reviewed') {
      // No second press gets past this: pressing s must never turn a flagged box into a label.
      var stacked = pendingDupDrops(dupGroups());
      var must = boxes().filter(function (b, i) { return stacked.indexOf(i) < 0 && mustDecide(b); }).length;
      if (must > 0) {
        say(must + ' box(es) the auto-labeler flagged (marked ?) need a decision: a accepts, r rejects, Shift+R rejects the rest.', true);
        select(selected >= 0 && mustDecide(boxes()[selected]) ? selected : firstReview(), true);
        return;
      }
      var n = reviewCount();
      if (n > 0 && !(guard && guard.frame === cur.frame && guard.count === n)) {
        guard = { frame: cur.frame, count: n };
        say(n + ' flagged hint box(es) (marked ?) are rejected on save unless you accept them; save again to reject them.', true);
        select(selected >= 0 && isReview(boxes()[selected]) ? selected : firstReview(), true);
        return;
      }
      // Then the check pass: accepted / rejected boxes not looked at (once per count).
      var u = uncheckedCount();
      if (n === 0 && u > 0 && !(guard && guard.frame === cur.frame && guard.unchecked === u)) {
        guard = { frame: cur.frame, count: 0, unchecked: u };
        say(u + ' accepted / rejected box(es) not double-checked yet (marked "to check"): a keeps, r rejects, or save again to save as they are.', true);
        var bs = boxes();
        if (!(selected >= 0 && isUnchecked(bs[selected]))) select(nextUndecided(-1), true);
        return;
      }
      // Then once per frame: missed objects are the commonest label error. The whole frame,
      // every box at full strength (no selection, no zoom; `x` keeps following on the next).
      if (askMissing && missingAsked !== cur.frame && !cur.labels.empty_confirmed) {
        missingAsked = cur.frame;
        selected = -1;
        view.z = 1; view.ox = 0; view.oy = 0;
        refresh();
        say(missingText());
        return;
      }
    }
    busy = true;
    var body = JSON.parse(JSON.stringify(cur.labels));   // keeps auto_run, bulk_accepted, ...
    // A pending box stacked on a better one must not be accepted by the save (on the copy: a
    // refused save loses nothing).
    var dropped = status === 'reviewed' ? pendingDupDrops(dupGroups()) : [];
    dropped.forEach(function (k) { markDuplicate(body.boxes[k]); });
    // Proposals of a "don't want" class are not shown: rejected rather than resolved by default.
    (body.overlap_suggestions || []).forEach(function (s) { delete s._done; });
    // The dataset check's suggestions go back as the page holds them (answers included).
    R.refineForSave(body);
    body.boxes.forEach(function (b) {
      delete b._checked;
      delete b._relabelled;
      delete b._n;
      if (b.state === 'pending' && b.source !== 'human' && isHiddenClass(b.label)) {
        b.state = 'rejected';
        b.reason = 'other_class';
      }
    });
    body.version = body.version || 1;
    body.status = status;
    body.empty_confirmed = !!cur.labels.empty_confirmed;
    body.classes_version = cur.labels.classes_version || '';
    // Stacks or overlaps still there after the save: the reviewer looked and kept them, so
    // auto=overlaps and the bulk reject leave the frame alone (as saved: hints go to rejected).
    // Whatever made a pair on this visit, an accepted dataset check suggestion too
    // (R.overlapsCheckedOnSave says why the save is the look).
    if (status === 'reviewed') {
      if (R.overlapsCheckedOnSave(body.boxes, classes, hiddenClasses)) body.overlaps_checked = true;
      else delete body.overlaps_checked;
    }
    var note = dropped.length ? ' Rejected ' + dropped.length + ' pending stacked duplicate(s).' : '';
    var frame = cur.frame, w = cur.width, h = cur.height;
    var url = '/v1/review/labels?frame=' + encodeURIComponent(frame);
    if (cur.revision != null) url += '&revision=' + encodeURIComponent(cur.revision);
    postJson(url, body).then(function (j) {
      dirty = false;
      if (window.PrismDraft) PrismDraft.drop(DRAFT);
      if (status === 'reviewed') { noteSaved(body); noteHistory(frame, body, w, h); }
      // After Back: on to the frame Back left (the counts refresh), else the queue's next.
      if (fwdStack.length) {
        say((j.message || 'Saved.') + note);
        loadNext(null, null, true);
        return loadFrame(fwdStack.pop(), 'forward');
      }
      return loadNext(frame, (j.message || 'Saved.') + note);
    }).catch(function (e) {
      if (e.conflict) showConflict(e.message);
      else say(e.message, true);
    }).then(function () { busy = false; });
  }

  // `n`: the next frame without saving; after Back, the frame Back left.
  function skipToNext() {
    if (nextComing()) return;
    if (!cur) return loadNext();
    if (!leaveOk()) return;
    if (fwdStack.length) {
      loadFrame(fwdStack.pop(), 'forward').then(function () { say(''); }).catch(function (e) { say(e.message, true); });
      return;
    }
    loadNext(cur.frame, '');
  }

  // `p`: the previous frame of this session, as saved (or not yet).
  function goBack() {
    if (!backStack.length) { say('No earlier frame in this session.'); return; }
    if (!leaveOk()) return;
    loadFrame(backStack.pop(), 'back').then(function () {
      say('Back to an earlier frame (' + (cur.saved ? 'saved as ' + cur.labels.status : 'not saved yet') + '): n returns, s saves it and returns.');
    }).catch(function (e) { say(e.message, true); });
  }

  // ---- session pace -------------------------------------------------------------------------

  function median(a) {
    var s = a.slice().sort(function (x, y) { return x - y; }), m = s.length >> 1;
    return s.length ? (s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2) : 0;
  }

  function acceptRate(fs) {
    var a = 0, r = 0;
    fs.forEach(function (f) { a += f.acc; r += f.rej; });
    return a + r ? a / (a + r) : null;
  }

  function seconds(fs) { return fs.map(function (f) { return f.s; }); }

  // A saved frame: its time (load to save, at most 10 min) and what saving made of its model boxes.
  function noteSaved(body) {
    var acc = 0, rej = 0;
    body.boxes.forEach(function (b) {
      // A duplicate or a too-small box is no verdict on an object, and nobody judged a dismissed
      // one: it would read as a false positive here.
      if (b.source === 'human' || b.reject_reason === 'duplicate' || b.reject_reason === 'too_small' || isDismissal(b.reason) || b.reason === 'parked_spot') return;
      var r = resolved(b);
      if (r === 'accepted') acc++; else if (r === 'rejected') rej++;
    });
    session.frames.push({ s: Math.min(600, (Date.now() - frameStart) / 1000), acc: acc, rej: rej });
    showSession();
  }

  // "Session 23 frames · 41 s/frame · 82% accepted · 37 min", a note when the last 10 frames went
  // much faster while accepting more (rubber-stamping), and a break note every 45 min.
  function showSession() {
    if (!sessionLine) return;
    var fs = session.frames, min = Math.floor((Date.now() - session.start) / 60000);
    sessionLine.textContent = '';
    if (!fs.length) return;
    var rate = acceptRate(fs);
    var parts = ['Session ' + fs.length + ' frame' + (fs.length === 1 ? '' : 's'), Math.round(median(seconds(fs))) + ' s/frame'];
    if (rate != null) parts.push(Math.round(rate * 100) + '% accepted');
    parts.push(min + ' min');
    sessionLine.appendChild(el('span', null, parts.join(' · ')));
    if (fs.length >= 15) {
      var last = fs.slice(-10), lr = acceptRate(last);
      if (median(seconds(last)) < 0.4 * median(seconds(fs)) && lr != null && rate != null && lr > rate) {
        sessionLine.appendChild(el('span', 'rv-chip warn', 'Going fast: check the small boxes'));
      }
    }
    if (Date.now() >= session.breakAt) {
      var note = el('span', 'rv-chip warn', 'Take a break? ');
      note.appendChild(button('Later', function () { session.breakAt = Date.now() + 45 * 60000; showSession(); }, 'secondary'));
      sessionLine.appendChild(note);
    }
  }
  setInterval(showSession, 30000);

  $('rv-save').addEventListener('click', function () { save('reviewed'); });
  $('rv-skip').addEventListener('click', function () { save('skipped'); });
  $('rv-next').addEventListener('click', skipToNext);
  if (backBtn) backBtn.addEventListener('click', goBack);
  $('rv-undo').addEventListener('click', doUndo);
  $('rv-accept-all').addEventListener('click', acceptProposals);
  if (likelyBar) likelyBar.addEventListener('click', acceptLikely);
  var infoBtn = $('rv-info-toggle');
  if (infoBtn) {
    infoBtn.addEventListener('click', function () {
      var open = $('review').classList.toggle('rv-info-open');
      infoBtn.textContent = open ? 'Hide' : 'Details';
      infoBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }
  $('rv-reject-rest').addEventListener('click', rejectRest);
  if (imgBtn) imgBtn.addEventListener('click', function () { cycleImg(); canvas.focus({ preventScroll: true }); });
  if (missingBox) {
    missingBox.checked = askMissing;
    missingBox.addEventListener('change', function () { askMissing = missingBox.checked; setPref('askMissing', askMissing); });
  }
  emptyBox.addEventListener('change', function () { confirmEmpty(emptyBox.checked); });
  // The filters in the page's URL (defaults left out), in the summary line and in the link to the
  // Auto-label tab.
  var DEFAULT_FILTERS = { status: 'unreviewed', source: 'queue' };
  function filtersShown() {
    var p = filterParams(), q = new URLSearchParams();
    p.forEach(function (v, k) { if (DEFAULT_FILTERS[k] !== v) q.set(k, v); });
    var text = q.toString();
    try { history.replaceState(null, '', location.pathname + (text ? '?' + text : '') + location.hash); } catch (e) { /* not kept */ }
    var link = $('rv-autolabel-link');
    if (link) link.href = '/train/autolabel?' + p.toString();
    var summary = $('rv-filter-summary');
    if (!summary) return;
    var f = filtersForm.elements;
    var name = function (k) { var e = f[k]; return e && e.options ? e.options[e.selectedIndex].text : ''; };
    var parts = [name('status'), name('source')];
    if (f.scene.value) parts.push(f.scene.value);
    if (f.camera.value) parts.push('camera ' + f.camera.value);
    if (f.label.value) parts.push(f.label.value.split(',').join(', '));
    if (f.bucket.value) parts.push('bucket ' + f.bucket.value);
    if (f.auto.value) parts.push(name('auto'));
    if (f.refine_kind && R.sendsFilter('refine_kind', f.refine_kind.value, f.auto.value)) parts.push(name('refine_kind'));
    summary.textContent = parts.join(', ');
  }
  // The kind of dataset check suggestion: offered with that Auto filter only.
  function showRefineKind() {
    var f = filtersForm.elements, wrap = $('rv-refine-kind');
    if (wrap) wrap.hidden = !(f.auto && f.auto.value === 'refine');
  }
  // A link's filters: unknown values are left at the default. `auto=overlaps` alone opens what the
  // Auto filter's choice opens.
  function filtersFromUrl() {
    var q = new URLSearchParams(location.search), f = filtersForm.elements;
    ['bucket', 'scene', 'camera', 'label', 'auto', 'refine_kind', 'status', 'source'].forEach(function (k) {
      var v = q.get(k), e = f[k];
      if (!v || !e) return;
      if (k === 'camera' && !Array.prototype.some.call(e.options, function (o) { return o.value === v; })) {
        var o = el('option', null, v);
        o.value = v;
        e.appendChild(o);
      }
      var was = e.value;
      e.value = v;
      if (e.options && e.value !== v) e.value = was;
    });
    if (q.get('auto') === 'overlaps' && !q.has('status') && !q.has('source')) showOverlaps();
    if (q.get('auto') === 'refine' && !q.has('status') && !q.has('source')) showRefine();
    showRefineKind();
  }
  // A filter change with unsaved edits asks first; kept, the frame stays and the counts follow.
  function filterReload() {
    filtersShown();
    if (leaveOk()) { loadNext(); return; }
    loadNext(null, null, true);
    say('Kept this frame with its unsaved changes; the new filters apply from the next frame.');
  }
  var filterTimer = null;
  function filterChanged() {
    if (filterTimer) clearTimeout(filterTimer);
    filterTimer = setTimeout(function () { filterTimer = null; filterReload(); }, 200);
  }
  // Overlaps sit on reviewed frames too, and a reviewed frame need not be in the review queue: the
  // Auto filter's "overlapping boxes" opens every status and every collected frame (both stay
  // choosable afterwards).
  function showOverlaps() {
    var f = filtersForm.elements;
    f.auto.value = 'overlaps';
    f.status.value = 'any';
    f.source.value = 'all';
  }
  // The dataset check judges reviewed frames: "dataset check suggestions" opens every status and
  // every collected frame, as overlapping boxes does.
  function showRefine() {
    var f = filtersForm.elements;
    f.auto.value = 'refine';
    f.status.value = 'any';
    f.source.value = 'all';
  }
  filtersForm.addEventListener('change', function (ev) {
    // The class picker reports through its own onChange (one reload per change).
    if (ev.target && ev.target.closest && ev.target.closest('.cp')) return;
    if (ev.target && ev.target.name === 'auto' && ev.target.value === 'overlaps') {
      showOverlaps();
      say('Overlapping boxes: Status set to any and Frames to all collected, so reviewed frames show too.');
    }
    if (ev.target && ev.target.name === 'auto' && ev.target.value === 'refine') {
      showRefine();
      say('Dataset check suggestions: Status set to any and Frames to all collected, so reviewed frames show too.');
    }
    showRefineKind();
    filterReload();
  });
  filtersForm.addEventListener('submit', function (ev) { ev.preventDefault(); filterReload(); });
  // Closing or reloading the tab with unsaved edits asks (the browser's own prompt).
  window.addEventListener('beforeunload', function (ev) {
    if (cur && dirty) { ev.preventDefault(); ev.returnValue = ''; }
  });

  // A field that takes the key itself. A checkbox / radio does not (the "Confirmed empty" box
  // keeps focus after a click); a box's class select takes only its own navigation keys.
  function typing(t, k) {
    var tag = t && t.tagName;
    if (tag === 'INPUT') return !/^(checkbox|radio|button|submit|reset)$/.test(t.type);
    if (tag === 'SELECT' && t.classList.contains('bx-label')) {
      return /^(Arrow(Up|Down|Left|Right)|Enter| |Home|End|PageUp|PageDown|Tab)$/.test(k);
    }
    return tag === 'TEXTAREA' || tag === 'SELECT' || !!(t && t.isContentEditable);
  }

  // ---- keys: one list for the table under Panel options and the `?` overlay ----------------

  // [group, key, text]; a text of null is the class row (filled by fillKeyHint), a key of '' a note.
  var KEYS = [
    ['Decide', 'a', 'accept the selected box (keep a drawn one) and go to the next'],
    ['Decide', 'r', 'reject the selected box (delete a drawn one) and go to the next'],
    ['Decide', 't', 'real, parked: accept the selected box and remember its spot, so that auto-label runs accept it there (a VLM no stays flagged); not on drawn boxes'],
    ['Decide', '1-9, 0', null],
    ['Decide', 'u', 'reject the selected box as a duplicate of the box it overlaps (stacked, nested or partly overlapping)'],
    ['Decide', 'Shift+U', 'stacked boxes: keep the best of each stack, reject the others as duplicates'],
    ['Decide', 'y', 'overlap suggestion: accept the one on screen (drops the box marked drop?, as a duplicate; or keeps both as two objects). Doing anything else ignores it'],
    ['Decide', 'y (dataset check)', 'once no overlap suggestion is on screen, accept the dataset check\'s suggestion (strongest first): add the dashed box, take the other class or the tighter box, or keep a label no model found. Doing anything else leaves it open'],
    ['Decide', 'w', 'dataset check suggestion: no (not an object, keep the label or the box as they are, or dismiss "no model found"); remembered, the next check does not ask again'],
    ['Decide', 'd', 'unsure: a real object you cannot name for sure: kept out of training, its frame goes to held-out only, ignored by the benchmark (?! on the box); again: not unsure'],
    ['Decide', 'Shift+1-9', 'after a rejection, or on a rejected box: why it was wrong (optional; nothing there, insect / web, rain / snow, IR glare, reflection, part of the scene, other object, too small, duplicate). Too small is also a button in the decision card: a real object too small to label, no negative, treated like unsure'],
    ['Decide', 'c', 'accept likely: every likely (★) box on the frame; the others stay pending'],
    ['Decide', 'Shift+A', 'accept proposals: the pending boxes saving would accept (flagged ? boxes and hints stay pending)'],
    ['Decide', 'Shift+R', 'reject the rest: every box still undecided, the way to clear the unlikely boxes'],
    ['Decide', 'e', 'confirmed empty (rejects every box)'],
    ['Decide', 'Delete', 'delete a drawn box, reject a model box'],
    ['Decide', 'z', 'undo'],
    ['Decide', 's, Enter', 'save & next; flagged (?) boxes need a decision first, then it asks about boxes not checked yet and "Anything missing?" (Panel options)'],
    ['Decide', 'drag', 'on a handle: resize; inside the selected box: move; elsewhere: draw a box'],
    ['Decide', 'arrows', 'move the selected box 1 px (Shift: 10 px)'],
    ['Decide', '', 'The next box is the next likely (★) one, then the next flagged (?) one, then the next pending one, then each accepted / rejected box not checked yet (check pass, marked to check).'],
    ['Navigate', 'j / k', 'select the next / previous box'],
    ['Navigate', 'Esc', 'no box selected'],
    ['Navigate', 'n', 'next frame without saving (after Back: the frame you came from)'],
    ['Navigate', 'p', 'back to the previous frame of this session (as saved)'],
    ['Navigate', 'x', 'zoom to the selected box and follow the selection (the next box, the next frame) until f, the wheel or a pan'],
    ['Navigate', 'f', 'fit (zoom out)'],
    ['Navigate', 'wheel', 'zoom about the pointer (up to 8x)'],
    ['Navigate', 'right / middle drag, Space + drag', 'pan'],
    ['View', 'b', 'image (view only): auto, off, bright, bright + equalised; auto = bright + equalised on night / IR frames'],
    ['View', '[ / ]', 'image darker / brighter (gamma)'],
    ['View', 'Shift+0', 'image back to auto, gamma 1'],
    ['View', 'v', 'spotlight on / off: the frame dimmed outside the selected box, the other boxes faint'],
    ['View', 'h (hold)', 'hide the boxes'],
    ['View', 'l', 'labels on the boxes on / off'],
    ['View', 'i', 'show / put away the hidden boxes of this frame (low-score auto boxes and boxes at spots you rejected, dimmed; accepting one makes it a label)'],
    ['View', '?', 'this list (Esc closes it)']
  ];

  function keyRows(tbody, group, withId) {
    KEYS.forEach(function (k) {
      if (group && k[0] !== group) return;
      var tr = el('tr');
      tr.appendChild(el('td', null, k[1]));
      var td = el('td');
      if (k[2] == null) {
        td.appendChild(document.createTextNode('class 1-10'));
        var span = el('span', 'rv-key-classes');
        if (withId) span.id = 'rv-key-classes';
        td.appendChild(span);
        td.appendChild(document.createTextNode(': relabel the selected box (accepting it) and go to the next, else the class of new boxes; the bar\'s It\'s a… lists them all'));
      } else td.textContent = k[2];
      if (!k[1]) td.className = 'muted';
      tr.appendChild(td);
      tbody.appendChild(tr);
    });
  }

  function renderKeys() {
    var tbody = document.querySelector('.review-keys tbody');
    if (tbody) {
      tbody.textContent = '';
      ['Decide', 'Navigate', 'View'].forEach(function (g) {
        var tr = el('tr', 'rv-keys-group');
        var th = el('th', null, g);
        th.colSpan = 2;
        tr.appendChild(th);
        tbody.appendChild(tr);
        keyRows(tbody, g, true);
      });
    }
    keysDialog.textContent = '';
    // Close first: the dialog focuses it on open, and it should not scroll the list to its end.
    var head = el('div', 'rv-keys-head');
    head.appendChild(el('h3', null, 'Keyboard'));
    head.appendChild(button('Close', function () { keysDialog.close(); }, 'secondary'));
    keysDialog.appendChild(head);
    var cols = el('div', 'rv-keys-cols');
    ['Decide', 'Navigate', 'View'].forEach(function (g) {
      var col = el('section');
      col.appendChild(el('h4', null, g));
      var t = el('table', 'review-keys');
      var tb = el('tbody');
      keyRows(tb, g, false);
      t.appendChild(tb);
      col.appendChild(t);
      cols.appendChild(col);
    });
    keysDialog.appendChild(cols);
  }

  function openKeys() {
    if (keysDialog.open) return;
    if (keysDialog.showModal) keysDialog.showModal(); else keysDialog.setAttribute('open', '');
  }
  keysDialog.addEventListener('click', function (ev) { if (ev.target === keysDialog) keysDialog.close(); });
  keysDialog.addEventListener('close', function () { canvas.focus({ preventScroll: true }); });
  if (keysBtn) keysBtn.addEventListener('click', openKeys);
  renderKeys();

  document.addEventListener('keydown', function (ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    var t = ev.target;
    if (typing(t, ev.key)) return;
    // The Browse dialog is modal: its keys are not review shortcuts.
    if (document.querySelector('dialog[open]')) return;
    // Letters are case-blind (Caps Lock); Shift is read from shiftKey.
    var k = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
    if (t && (t.tagName === 'BUTTON' || t.tagName === 'SUMMARY' || t.tagName === 'INPUT') && (k === 'Enter' || k === ' ')) {
      // Held on a button, Enter clicks it again per repeat, and a second Save is "save anyway".
      if (ev.repeat && t.tagName === 'BUTTON') ev.preventDefault();
      return;
    }
    // A letter on a box's class select is a shortcut, never "jump to the class starting with it";
    // the frame takes the focus back.
    if (t && t.tagName === 'SELECT') {
      if (k.length === 1) ev.preventDefault();
      canvas.focus({ preventScroll: true });
    }
    // Shift+1-9: the reason of the box just rejected (or the selected rejected box).
    if (ev.shiftKey && /^Digit[1-9]$/.test(ev.code) && reasonTarget() >= 0) {
      setReason(Number(ev.code.slice(5)) - 1);
      ev.preventDefault();
      return;
    }
    var handled = true;
    if (k === ' ') {
      spaceDown = true;
      if (!gesture) setCursor('grab');
      // Space scrolls the page unless the pointer or the focus is on the frame.
      handled = overCanvas || document.activeElement === canvas;
    } else if (k === 'h') {
      if (!ev.repeat && !hideBoxes) { hideBoxes = true; draw(); }
    } else if (k === 'l') toggleTags();
    else if (k === 'v') toggleSpotlight();
    else if (k === 'x') zoomToSelected();
    else if (k === 'f') fit();
    else if (k === 'b') cycleImg();
    else if (k === '[' || k === ']') stepGamma(k === ']' ? 1 : -1);
    else if (ev.code === 'Digit0' && ev.shiftKey) setImgView('auto', 1);
    else if (k === 'p') goBack();
    else if (k === 'd') toggleUnsure();
    else if (k === 't') parkSelected();
    else if (k === 'u') { if (ev.shiftKey) rejectDuplicates(); else rejectDupSelected(); }
    // `y`: the overlap suggestion on screen, else the dataset check's (R.suggestionKey); `w`: no
    // to the dataset check's, else nothing (the key stays the browser's). Once per press: a held
    // key ('held') answers nothing more.
    else if (k === 'y' || k === 'w') {
      var route = suggestionRoute(k, ev.repeat);
      if (route === 'accept' || route === 'dismiss') answerRefine(route === 'accept');
      else if (route === 'overlap') acceptOverlap();
      else if (route === 'none') handled = false;
    }
    else if (k === 'c') { if (!ev.shiftKey) acceptLikely(); }
    else if (k === 'i') toggleDismissed();
    else if (k === '?') openKeys();
    else if (k === 'ArrowLeft' || k === 'ArrowRight' || k === 'ArrowUp' || k === 'ArrowDown') {
      var step = ev.shiftKey ? 10 : 1;
      handled = selected >= 0 && nudge(
        k === 'ArrowLeft' ? -step : (k === 'ArrowRight' ? step : 0),
        k === 'ArrowUp' ? -step : (k === 'ArrowDown' ? step : 0));
    } else if (k === 'a') {
      if (ev.shiftKey) acceptProposals(); else acceptSelected();
    } else if (k === 'r') {
      if (ev.shiftKey) rejectRest(); else rejectSelected();
    }
    // Held keys: `n` and `s` once per press (each costs a walk of the folder, and a second `s` is
    // what passes the save's "are you sure" steps); the arrows above keep repeating.
    else if (k === 'n') { if (!ev.repeat) skipToNext(); }
    else if (k === 'e') confirmEmpty(!(cur && cur.labels.empty_confirmed));
    else if (k === 'z') doUndo();
    else if (k === 's' || k === 'Enter') { if (!ev.repeat) save('reviewed'); }
    else if (k === 'j') select(stepShown(1), true);
    else if (k === 'k') select(stepShown(-1), true);
    else if (k === 'Delete' || k === 'Backspace') {
      if (selected >= 0) {
        if (boxes()[selected].source === 'human') removeBox(selected);
        else rejectWith(selected, null);
      }
    } else if (k === 'Escape') {
      // A two-tap box waiting for its corners is cancelled first (Draw itself stays on).
      if (taps) cancelTaps(true); else select(-1);
    }
    else if (/^[0-9]$/.test(k)) pickClass(k === '0' ? 9 : Number(k) - 1);
    else handled = false;
    if (handled) ev.preventDefault();
  });

  document.addEventListener('keyup', function (ev) {
    if ((ev.key === 'h' || ev.key === 'H') && hideBoxes) { hideBoxes = false; draw(); }
    if (ev.key === ' ' && spaceDown) { spaceDown = false; if (!gesture) hoverCursor(lastPointer); }
  });
  window.addEventListener('blur', function () {
    var redraw = hideBoxes;
    hideBoxes = false; spaceDown = false;
    if (redraw) draw();
  });

  // ---- follow a job, bulk tools -------------------------------------

  function report(box, lines, warnings) {
    box.textContent = '';
    lines.forEach(function (l) { box.appendChild(el('div', null, l)); });
    if (warnings && warnings.length) {
      var ul = el('ul', 'warn-list');
      warnings.slice(0, 50).forEach(function (w) { ul.appendChild(el('li', null, w)); });
      if (warnings.length > 50) ul.appendChild(el('li', null, '... ' + (warnings.length - 50) + ' more'));
      box.appendChild(ul);
    }
  }

  var alStatus = $('rv-al-status');
  var alProgress = $('rv-al-progress');
  var polling = null;   // {id, box, bar, after, loop}: the task followed

  function describeTask(t) {
    if (t.state === 'queued') return 'Queued: ' + (t.title || 'auto-label') + ' (waiting for the queue).';
    var s = t.stage ? String(t.stage).replace(/_/g, ' ') : 'starting';
    return (t.title ? t.title + ': ' : '') + s + (t.steps ? ' (' + t.step + '/' + t.steps + ')' : '') +
      (t.percent != null ? ', ' + t.percent + '%' : '') + (t.detail ? ': ' + t.detail : '');
  }

  function stopPolling() {
    if (polling) polling.loop.stop();
    polling = null;
  }

  // Task records: task `id` (a list of one, none when the server has no such task), else every
  // queued, running or paused one. From GET /v1/train/jobs (task=<id> / tasks=busy: the records
  // without log or result); a server whose answer lacks the field is asked on GET /v1/training
  // from then on.
  var jobsHasTasks = true;
  function taskRecords(id, asPoll) {
    // Asked from a poll (here, or a fallback chained after one's answer): a poll throughout.
    var poll = asPoll || (window.PrismPoll && window.PrismPoll.inPoll && window.PrismPoll.inPoll());
    var get = poll && window.PrismPoll.fetch ? window.PrismPoll.fetch : fetch;
    var json = function (url) {
      return get(url, { headers: { Accept: 'application/json' }, cache: 'no-store' }).then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      });
    };
    if (!jobsHasTasks) {
      return json('/v1/training').then(function (data) {
        return (data.tasks || []).filter(function (t) { return id ? t.id === id : true; });
      });
    }
    var key = id ? 'task' : 'tasks';
    return json('/v1/train/jobs?' + (id ? 'task=' + encodeURIComponent(id) : 'tasks=busy')).then(function (j) {
      if (!j || !(key in j)) { jobsHasTasks = false; return taskRecords(id, poll); }
      return id ? (j.task ? [j.task] : []) : (j.tasks || []);
    });
  }

  // Chained, never an interval (no two requests at once), no request while the tab is hidden, one
  // at once when it shows again. `run()` resolves false to stop.
  function loop(run, period) {
    if (window.PrismPoll) {
      var p = window.PrismPoll.create({ run: run, period: period, more: function (go) { return go !== false; } });
      p.start();
      return p;
    }
    // An older live.js: the same loop without the back-off.
    var timer = null, inFlight = false, done = false;
    var clear = function () { if (timer !== null) { clearTimeout(timer); timer = null; } };
    var go = function () {
      clear();
      if (done || inFlight || document.hidden) return;
      inFlight = true;
      run().then(function (more) { if (more === false) done = true; }, function () {}).then(function () {
        inFlight = false;
        if (!done && !document.hidden) timer = setTimeout(go, period);
      });
    };
    document.addEventListener('visibilitychange', function () { if (document.hidden) clear(); else go(); });
    go();
    return { stop: function () { done = true; clear(); } };
  }

  // Follow task `id`; `box` and `bar` show it (default: the line above the frame), `after(t)` runs when
  // it ends.
  function pollTask(id, box, bar, after) {
    stopPolling();
    var me = { id: id, box: box || alStatus, bar: bar || alProgress, after: after };
    polling = me;
    me.loop = loop(function () { return tick(me); }, 3000);
  }

  function tick(me) {
    return taskRecords(me.id).then(function (list) {
      if (polling !== me) return false;
      var t = list.filter(function (x) { return x.id === me.id; })[0];
      if (!t) return true;
      var box = me.box, bar = me.bar, after = me.after;
      if (t.state === 'queued' || t.state === 'running' || t.state === 'paused') {
        report(box, [describeTask(t)]);
        bar.hidden = false;
        bar.value = t.percent || 0;
        return true;
      }
      stopPolling();
      bar.hidden = true;
      if (t.state === 'done') report(box, ['Done: ' + (t.summary || t.title || 'auto-label finished')]);
      else if (t.state === 'failed') report(box, ['Failed: ' + (t.error || 'see Jobs on the Auto-label tab.')]);
      else report(box, [t.summary ? 'Cancelled: ' + t.summary : 'Cancelled.']);
      box.className = 'review-report ' + (t.state === 'done' ? 'ok-text' : 'error-text');
      // Asked for by the poll's answer, not the reviewer.
      background(function () {
        if (after) after(t);
        reloadQueue();
      });
      return false;
    });
  }

  // A job queued earlier (here or on the Auto-label tab): follow it.
  if (alStatus) {
    taskRecords(null)
      .then(function (list) {
        if (polling) return;
        var busyTasks = list.filter(function (t) {
          return (t.kind === 'autolabel' || t.kind === 'overlaps') && (t.state === 'queued' || t.state === 'running' || t.state === 'paused');
        });
        var t = busyTasks.filter(function (x) { return x.state === 'running'; })[0] || busyTasks[0];
        if (t && t.kind === 'overlaps') followOverlaps(t.id);
        else if (t) pollTask(t.id);
      })
      .catch(function () {});
  }

  var bulkBtn = $('rv-accept-auto');
  if (bulkBtn) {
    bulkBtn.addEventListener('click', function () {
      var box = $('rv-accept-auto-msg');
      var body = {};
      ['bucket', 'scene', 'camera', 'label', 'source'].forEach(function (k) {
        var e = filtersForm.elements[k];
        if (e && e.value) body[k] = e.value;
      });
      body.dry_run = true;
      box.className = 'review-report';
      report(box, ['Counting the frames...']);
      bulkBtn.disabled = true;
      postJson('/v1/review/accept-auto', body).then(function (j) {
        var n = j.eligible || 0;
        if (!n) { report(box, [j.message || 'No unreviewed frame with only auto: accept boxes matches the filters.']); return null; }
        if (!window.confirm('Accept ' + n + ' frame(s) whose boxes the auto-labeler marked accept? ' +
          'They are saved as reviewed: those boxes accepted, unconfirmed hint boxes rejected. ' +
          'Frames with a box flagged for review are left alone.')) {
          report(box, ['Nothing accepted.']);
          return null;
        }
        body.dry_run = false;
        return postJson('/v1/review/accept-auto', body).then(function (r) {
          var failed = r.failed || [];
          report(box, [r.message || ('Accepted ' + (r.accepted || 0) + ' frame(s).')],
            failed.map(function (f) { return typeof f === 'string' ? f : JSON.stringify(f); }));
          box.className = 'review-report ' + (failed.length ? 'error-text' : 'ok-text');
          return reloadQueue();
        });
      }).catch(function (e) {
        box.className = 'review-report error-text';
        report(box, [e.message]);
      }).then(function () { bulkBtn.disabled = false; });
    });
  }

  // Save the frames spot edits settled (nothing pending left): dry run, ask, save.
  var repeatsBtn = $('rv-accept-repeats');
  if (repeatsBtn) {
    repeatsBtn.addEventListener('click', function () {
      var box = $('rv-accept-repeats-msg') || msg;
      var body = {};
      ['bucket', 'scene', 'camera', 'label', 'source'].forEach(function (k) {
        var e = filtersForm.elements[k];
        if (e && e.value) body[k] = e.value;
      });
      body.dry_run = true;
      box.className = 'review-report';
      report(box, ['Counting the frames...']);
      repeatsBtn.disabled = true;
      postJson('/v1/review/accept-repeats', body).then(function (j) {
        var n = j.eligible || 0;
        if (!n) { report(box, ['No repeat-only frame matches the filters (a frame with a box still to decide never is).']); return null; }
        if (!window.confirm('Save ' + n + ' repeat-only frame(s) as reviewed? Their boxes were all settled by "this spot on all frames"; ' +
          'nothing on them is pending.')) {
          report(box, ['Nothing saved.']);
          return null;
        }
        body.dry_run = false;
        return postJson('/v1/review/accept-repeats', body).then(function (r) {
          var failed = r.failed || [];
          report(box, [r.message || ('Saved ' + (r.accepted || 0) + ' frame(s).')], failed);
          box.className = 'review-report ' + (failed.length ? 'error-text' : 'ok-text');
          return reloadQueue();
        });
      }).catch(function (e) {
        box.className = 'review-report error-text';
        report(box, [e.message]);
      }).then(function () { repeatsBtn.disabled = false; });
    });
  }

  var rejectBtn = $('rv-reject-low');
  if (rejectBtn) {
    rejectBtn.addEventListener('click', function () {
      var box = $('rv-reject-low-msg');
      var score = parseFloat($('rv-reject-low-score').value);
      box.className = 'review-report';
      if (!(score >= 0.01 && score <= 1)) {
        box.className = 'review-report error-text';
        report(box, ['Enter a score between 0.01 and 1.']);
        return;
      }
      var body = { min_score: score };
      ['bucket', 'scene', 'camera', 'label', 'source'].forEach(function (k) {
        var e = filtersForm.elements[k];
        if (e && e.value) body[k] = e.value;
      });
      body.dry_run = true;
      report(box, ['Counting the boxes...']);
      rejectBtn.disabled = true;
      postJson('/v1/review/reject-low', body).then(function (j) {
        var n = j.boxes || 0;
        if (!n) { report(box, [j.message || ('No pending auto box below ' + score + ' on the unreviewed frames the filters select.')]); return null; }
        if (!window.confirm('Reject ' + n + ' box(es) on ' + (j.eligible || 0) + ' frame(s): the pending auto boxes ' +
          'scoring below ' + score + '? They are kept as hard negatives (reason low_score); the frames stay ' +
          'unreviewed and a reviewer can still change them per frame.')) {
          report(box, ['Nothing rejected.']);
          return null;
        }
        body.dry_run = false;
        return postJson('/v1/review/reject-low', body).then(function (r) {
          var failed = r.failed || [];
          report(box, [r.message || ('Rejected ' + (r.boxes || 0) + ' box(es).')],
            failed.map(function (f) { return typeof f === 'string' ? f : JSON.stringify(f); }));
          box.className = 'review-report ' + (failed.length ? 'error-text' : 'ok-text');
          return reloadQueue();
        });
      }).catch(function (e) {
        box.className = 'review-report error-text';
        report(box, [e.message]);
      }).then(function () { rejectBtn.disabled = false; });
    });
  }

  // `r` = reviewed_with_unwanted {frames, boxes, classes}: asks whether to reject the accepted
  // boxes of unwanted classes on reviewed frames too (POST /v1/review/reject-unwanted-reviewed)
  // and appends the outcome to `box`. Resolves either way.
  function askReviewedUnwanted(r, box) {
    if (!r || !(r.frames > 0)) return Promise.resolve();
    var text = r.frames + ' reviewed frame(s) have ' + r.boxes + ' accepted box(es) of ' +
      (r.classes || []).join(', ') + '. Reject those too? They are kept as evidence (reason other_class) ' +
      'and the frames stay reviewed; Cancel keeps them as training labels.';
    if (!window.confirm(text)) {
      box.appendChild(el('div', null, 'Reviewed frames left as they are (' + r.frames + ' with boxes of ' +
        (r.classes || []).join(', ') + ').'));
      return Promise.resolve();
    }
    return postJson('/v1/review/reject-unwanted-reviewed', {}).then(function (j) {
      box.appendChild(el('div', null, j.message || ('Rejected ' + (j.boxes || 0) + ' box(es).')));
      if (j.failed && j.failed.length) box.appendChild(el('div', 'error-text', j.failed.join('; ')));
      return reloadQueue();
    }).catch(function (e) {
      box.appendChild(el('div', 'error-text', e.message));
    });
  }

  var otherBtn = $('rv-reject-other');
  if (otherBtn) {
    otherBtn.addEventListener('click', function () {
      var box = $('rv-reject-other-msg');
      var body = {};
      ['bucket', 'scene', 'camera', 'label', 'source'].forEach(function (k) {
        var e = filtersForm.elements[k];
        if (e && e.value) body[k] = e.value;
      });
      body.dry_run = true;
      box.className = 'review-report';
      report(box, ['Counting the boxes...']);
      otherBtn.disabled = true;
      postJson('/v1/review/reject-other-class', body).then(function (j) {
        var n = j.boxes || 0;
        if (!n) {
          report(box, [j.message || 'No pending box of an unwanted class (hidden, or not in the class list) on the unreviewed frames the filters select.']);
          return askReviewedUnwanted(j.reviewed_with_unwanted, box);
        }
        var renamed = j.relabelled || 0;
        var text = 'Reject ' + (n - renamed) + ' pending model / auto box(es) of unwanted classes (hidden, or not in ' +
          'the class list) on ' + (j.eligible || 0) + ' frame(s)? They are kept as evidence (reason other_class); ' +
          'the frames stay unreviewed and a reviewer can still change them per frame.';
        if (renamed) {
          text += ' ' + renamed + ' box(es) whose label is a class of the list under another spelling ' +
            '(e.g. pottedplant) are renamed to the list\'s spelling instead.';
        }
        if (!window.confirm(text)) {
          report(box, ['Nothing rejected.']);
          return null;
        }
        body.dry_run = false;
        return postJson('/v1/review/reject-other-class', body).then(function (r) {
          var failed = r.failed || [];
          report(box, [r.message || ('Changed ' + (r.boxes || 0) + ' box(es).')],
            failed.map(function (f) { return typeof f === 'string' ? f : JSON.stringify(f); }));
          box.className = 'review-report ' + (failed.length ? 'error-text' : 'ok-text');
          return reloadQueue().then(function () { return askReviewedUnwanted(r.reviewed_with_unwanted, box); });
        });
      }).catch(function (e) {
        box.className = 'review-report error-text';
        report(box, [e.message]);
      }).then(function () { otherBtn.disabled = false; });
    });
  }

  // The overlap check: count, confirm, queue, follow. It writes suggestions only; the result line
  // offers the queue of overlapping boxes.
  function followOverlaps(id) {
    var box = $('rv-suggest-msg');
    pollTask(id, box, $('rv-suggest-progress'), function (t) {
      if (t.state !== 'done') return;
      box.appendChild(button('Show the overlapping frames', function () {
        showOverlaps();
        filterReload();
      }));
    });
  }

  var suggestBtn = $('rv-suggest-overlaps');
  if (suggestBtn) {
    suggestBtn.addEventListener('click', function () {
      var box = $('rv-suggest-msg');
      var body = {};
      ['bucket', 'scene', 'camera', 'label'].forEach(function (k) {
        var e = filtersForm.elements[k];
        if (e && e.value) body[k] = e.value;
      });
      var large = $('rv-suggest-large');
      if (large && large.checked) body.allow_large = true;
      var askAgain = $('rv-suggest-again');
      if (askAgain && askAgain.checked) body.again = true;
      body.dry_run = true;
      box.className = 'review-report';
      report(box, ['Counting the pairs...']);
      suggestBtn.disabled = true;
      postJson('/v1/review/suggest-overlaps', body).then(function (j) {
        if (!j.pairs) {
          report(box, [j.message || 'No open overlap without a suggestion on these frames.']);
          return null;
        }
        if (!window.confirm('Check ' + j.pairs + ' overlapping pair(s) on ' + j.frames + ' frame(s), reviewed frames included' +
          (body.again ? ', answered pairs asked again' : '') + ', ' +
          'with RF-DETR L (' + j.crops + ' crops, on the training queue, about ' + Math.max(1, Math.ceil(j.crops * 0.08 / 60)) + ' min)? ' +
          'It writes suggestions only: no box changes until you confirm one here.')) {
          report(box, ['Nothing queued.']);
          return null;
        }
        body.dry_run = false;
        return postJson('/v1/review/suggest-overlaps', body).then(function (r) {
          report(box, [r.message || 'Queued.']);
          if (r.id) followOverlaps(r.id);
        });
      }).catch(function (e) {
        box.className = 'review-report error-text';
        report(box, [e.message]);
      }).then(function () { suggestBtn.disabled = false; });
    });
  }

  // Stacked boxes span reviewed frames too; the Frames filter does not narrow it (a reviewed frame
  // need not be in the review queue). Overlaps that are no stack stay a reviewer's: the server's
  // message says how many and how to list them.
  var stackedBtn = $('rv-reject-stacked');
  if (stackedBtn) {
    stackedBtn.addEventListener('click', function () {
      var box = $('rv-reject-stacked-msg');
      var body = {};
      ['bucket', 'scene', 'camera', 'label'].forEach(function (k) {
        var e = filtersForm.elements[k];
        if (e && e.value) body[k] = e.value;
      });
      body.dry_run = true;
      box.className = 'review-report';
      report(box, ['Counting the boxes...']);
      stackedBtn.disabled = true;
      postJson('/v1/review/reject-stacked', body).then(function (j) {
        var n = j.boxes || 0;
        if (!n) {
          report(box, [j.message || 'No stacked duplicates on the frames these filters select.']);
          return null;
        }
        if (!window.confirm('Reject ' + n + ' stacked duplicate box(es) on ' + (j.eligible || 0) + ' frame(s), reviewed frames ' +
          'included? The best box of each stack stays. Frames stay reviewed and the boxes are kept as rejected evidence ' +
          '(reason duplicate, left out of the hard negatives); drawn boxes are never rejected.')) {
          report(box, ['Nothing rejected.']);
          return null;
        }
        body.dry_run = false;
        return postJson('/v1/review/reject-stacked', body).then(function (r) {
          var failed = r.failed || [];
          report(box, [r.message || ('Rejected ' + (r.boxes || 0) + ' box(es).')],
            failed.map(function (f) { return typeof f === 'string' ? f : JSON.stringify(f); }));
          box.className = 'review-report ' + (failed.length ? 'error-text' : 'ok-text');
          histories = {};   // the boxes rejected were decisions of these cameras
          return reloadQueue();
        });
      }).catch(function (e) {
        box.className = 'review-report error-text';
        report(box, [e.message]);
      }).then(function () { stackedBtn.disabled = false; });
    });
  }

  // The undo of a re-label that reopened reviewed frames: back to reviewed when nothing is pending.
  var restoreBtn = $('rv-restore-reopened');
  if (restoreBtn) {
    restoreBtn.addEventListener('click', function () {
      var box = $('rv-restore-reopened-msg');
      var body = {};
      ['bucket', 'scene', 'camera', 'label', 'source'].forEach(function (k) {
        var e = filtersForm.elements[k];
        if (e && e.value) body[k] = e.value;
      });
      body.dry_run = true;
      box.className = 'review-report';
      report(box, ['Counting the frames...']);
      restoreBtn.disabled = true;
      postJson('/v1/review/restore-reopened', body).then(function (j) {
        var n = j.eligible || 0;
        if (!n) {
          report(box, ['No reopened frame with nothing left pending among the frames the filters select.']);
          return null;
        }
        if (!window.confirm('Put ' + n + ' reopened frame(s) back to reviewed? Their boxes stay as they are, and they ' +
          'return to exports. Frames with a pending box are not touched.')) {
          report(box, ['Nothing restored.']);
          return null;
        }
        body.dry_run = false;
        return postJson('/v1/review/restore-reopened', body).then(function (r) {
          var failed = r.failed || [];
          report(box, [r.message || ('Restored ' + (r.restored || 0) + ' frame(s).')],
            failed.map(function (f) { return typeof f === 'string' ? f : JSON.stringify(f); }));
          box.className = 'review-report ' + (failed.length ? 'error-text' : 'ok-text');
          return reloadQueue();
        });
      }).catch(function (e) {
        box.className = 'review-report error-text';
        report(box, [e.message]);
      }).then(function () { restoreBtn.disabled = false; });
    });
  }

  // ---- folder under review -------------------------------------------------------------------

  function setFolder(dir) {
    if (!leaveOk()) return;
    var box = $('rv-folder-msg');
    box.textContent = 'Switching...';
    box.className = 'form-msg';
    postJson('/v1/review/folder', { dir: dir }).then(function () {
      dirty = false;   // asked above: no second prompt on the reload
      location.reload();
    }).catch(function (e) { box.textContent = e.message; box.className = 'form-msg error-text'; });
  }

  $('rv-folder').addEventListener('submit', function (ev) {
    ev.preventDefault();
    setFolder(ev.target.elements.dir.value.trim());
  });
  if ($('rv-folder-reset')) {
    $('rv-folder-reset').addEventListener('click', function () { setFolder(''); });
  }

  // ---- class pickers ------------------------------------------------------------------------

  function initPickers() {
    if (!CP) return;
    var entries = buildEntries();
    var labelHidden = $('rv-label');
    var fb = $('rv-label-fallback');
    var holder = $('rv-label-picker');
    if (holder && labelHidden) {
      var reload = filterChanged;
      labelPicker = CP.create(holder, {
        entries: entries, selected: labelHidden.value ? labelHidden.value.split(',') : [], mode: 'compact', label: 'Class', emptyText: 'any',
        onChange: function (sel) { labelHidden.value = sel.join(','); reload(); }
      });
    }
    var showHolder = $('rv-show-picker');
    if (showHolder) {
      showPicker = CP.create(showHolder, {
        entries: entries, selected: [], mode: 'compact', emptyText: 'all classes',
        onChange: function (sel) { shownLabels = sel; refresh(); }
      });
    }
  }

  fillKeyHint();
  if (CP) {
    CP.fetchCatalog().then(function (c) {
      catalog = c;
      if (Array.isArray(c.classes)) classes = c.classes;
      hiddenClasses = Array.isArray(c.hidden) ? c.hidden : [];
      fillClassSelect();
      fillKeyHint();
    }).catch(function () { catalog = null; }).then(function () {
      initPickers();
      refresh();
    });
  }
  filtersFromUrl();
  filtersShown();
  // A draft (above) opens its frame; the queue's counts load beside it. Gone: the queue as usual.
  var draft0 = window.PrismDraft ? PrismDraft.load(DRAFT) : null;
  if (draft0 && typeof draft0.id === 'string' && draft0.id) {
    draftWaiting = true;
    loadNext(null, null, true);
    loadFrame(draft0.id).then(function () { draftWaiting = false; }, function () {
      draftWaiting = false;
      PrismDraft.drop(DRAFT);
      say('Your unsaved edits from before the page was closed were dropped: their frame is gone.', true);
      loadNext();
    });
  } else {
    loadNext();
  }
})();
