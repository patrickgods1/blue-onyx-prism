// Review page (/review): one frame at a time from GET /v1/review/next, its boxes from
// GET /v1/review/labels, drawn on a canvas; decisions saved with POST /v1/review/labels
// (with the loaded `revision`; a 409 means someone else saved the frame meanwhile).
// Filters (bucket, scene, camera, class, auto, status, source) go to /v1/review/next as query
// params. The canvas edits boxes (draw, select, move, resize, nudge) under a zoom / pan view.
// The Auto-label card queues a job (POST /training/autolabel as JSON) and follows it on
// GET /v1/training; "Accept auto: accept frames" calls POST /v1/review/accept-auto and
// "Reject auto boxes below" POST /v1/review/reject-low and "Reject boxes of unwanted classes"
// POST /v1/review/reject-other-class (all three: dry run, confirm, run).
// Export / import / classes / folder forms post JSON to /v1/review/{export,import,classes,folder};
// a new folder reloads the page.
// Classes: GET /v1/review/class-catalog feeds the class pickers (class-picker.js): the Class
// filter (label=a,b), "Show boxes of" (display only), the auto-label form (repeated `classes`
// fields; a class outside the list is appended after a confirm) and the Classes card (list in
// order, "don't want (hide + reject)" per class = collect.hidden_classes). Unwanted classes
// (hidden, or not in the list) are what the reviewer does not review: pickers, the new-box class
// select and the 1-0 keys leave the hidden ones out, the canvas / box list / stepping skip boxes
// of hidden classes and rejected boxes of unlisted ones (they stay in the file; a save rejects
// pending proposals of hidden classes), and the server rejects their pending boxes when a class
// becomes unwanted. When a class save, the reject button or the auto-label form (unticked listed
// classes: "Hide and auto-reject ...?") leaves reviewed frames with accepted boxes of unwanted
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
  // The keys overlay.
  var keysDialog = $('rv-keys-dialog');
  if (!keysDialog) {
    keysDialog = document.createElement('dialog');
    keysDialog.id = 'rv-keys-dialog';
    keysDialog.className = 'rv-keys-dialog';
    document.body.appendChild(keysDialog);
  }

  var cur = null;      // answer of GET /v1/review/labels (with `revision`)
  var img = null;       // the frame image
  var selected = -1;    // index into cur.labels.boxes
  var undo = [];        // snapshots of cur.labels
  var dirty = false;
  var gesture = null;   // the pointer gesture in progress (see the canvas input section)
  var busy = false;
  var view = { z: 1, ox: 0, oy: 0, follow: false };   // zoom, the frame point at the canvas's top
                                                      // left; `follow`: `x` zooms to each box
  var hideBoxes = false;               // `h` held
  var showTags = true;                 // `l` toggles
  var spotlight = pref('spotlight', true);   // `v`: veil the frame and dim the other boxes
  var wideSide = pref('wide', false);        // a larger close-up (wider side panel)
  var hoverIdx = -1;                   // the box under the pointer (its tag shows in full)
  var classGrid = false;               // the decision bar's "It's a..." class buttons open
  var spaceDown = false;               // Space held: left-drag pans
  var overCanvas = false;
  var guard = null;                    // {frame, count}: the save that warned about review boxes
  var emptyBackup = null;              // {frame, boxes}: the boxes before "Confirmed empty"
  var upcoming = [];                   // the next frames of the queue (prefetch)
  var prefetchImg = null;
  var seenLabels = [];                 // classes the queue saw (Class filter)
  var lastNudge = null;                // {i, t}: arrow nudges in a row share one undo step
  var labelPicker = null, showPicker = null, alPicker = null, classesPicker = null;
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
  var dups = { groups: [], of: {} };   // stacked boxes (dupGroups), as of the last render

  var COLORS = {
    pending: '#e8d000', review: '#ffa000', hint: '#f08c00', accepted: '#2e9e4f',
    added: '#1f6fe0', rejected: '#e53935'
  };
  var MAX_ZOOM = 8;
  var HANDLE_CSS = 5;   // half size of a resize handle, CSS pixels
  var CLICK_CSS = 4;    // a pointer that moves less than this is a click
  var MIN_BOX = 3;      // frame pixels

  function say(text, isError) {
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
    Array.prototype.forEach.call(filtersForm.elements, function (el) {
      if (el.name && el.value) p.set(el.name, el.value);
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

  // What saving makes of a box (mirrors labels::resolve_pending).
  function resolved(b) {
    if (b.state !== 'pending') return b.state;
    if (b.source === 'human') return 'added';
    if (b.source === 'auto') return 'accepted';
    if (b.origin === 'prediction') return 'accepted';
    return (b.triage === 'accept' && !b.auto_label) ? 'accepted' : 'rejected';
  }

  // A pending model box saving rejects unless accepted (suppressed, below threshold, ...).
  function isHint(b) { return b.state === 'pending' && resolved(b) === 'rejected'; }

  // A pending box the auto-labeler flagged: triage `review`, or it names another class (never
  // one of an unwanted class: it is not shown).
  function isReview(b) {
    return b.state === 'pending' && !isUnwanted(b) &&
      (b.triage === 'review' || (!!b.auto_label && b.auto_label !== b.label));
  }

  // What a box shows as: the state, with pending split by review / what saving will make of it.
  function look(b) {
    if (b.state === 'pending') return isReview(b) ? 'review' : (isHint(b) ? 'hint' : 'pending');
    return b.state;
  }

  // The box `d` (1 or -1) steps from the selected one, wrapping, past the unwanted boxes (not
  // shown); -1 when none is shown.
  function stepShown(d) {
    var bs = boxes();
    for (var k = 1; k <= bs.length; k++) {
      var j = ((selected < 0 && d < 0 ? 0 : selected) + d * k + bs.length * 2) % bs.length;
      if (!isUnwanted(bs[j])) return j;
    }
    return -1;
  }

  function reviewCount() { return boxes().filter(isReview).length; }

  function firstReview() {
    var bs = boxes();
    for (var i = 0; i < bs.length; i++) if (isReview(bs[i])) return i;
    return -1;
  }

  // A decided box (accepted, rejected, drawn) not looked at on this visit of the frame: the
  // check pass steps through them once nothing is pending. `_checked` is page-only (not saved).
  function isUnchecked(b) {
    return b.state !== 'pending' && !b._checked && isShown(b);
  }

  function uncheckedCount() { return boxes().filter(isUnchecked).length; }

  function undecidedCount() {
    return boxes().filter(function (b) { return b.state === 'pending' && !isUnwanted(b); }).length;
  }

  // The box to look at after `i` (wrapping; -1 for the first): the next review box, else the
  // next pending one, else (the check pass) the first unchecked decided one, else -1.
  function nextUndecided(i) {
    var bs = boxes();
    var tests = [isReview, function (b) { return b.state === 'pending' && !isUnwanted(b); }];
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
    var u = undecidedCount(), c = uncheckedCount();
    if (u) say(u + ' box(es) still undecided.');
    else if (c) say('Check pass: ' + c + ' accepted / rejected box(es) left to double-check (a keep / accept, r reject).');
    else say('Every box decided and checked: s saves.');
  }

  function fmtScore(s) { return (typeof s === 'number') ? s.toFixed(2) : ''; }

  // The classes Review offers: the list minus the hidden ones.
  function visible() { return classes.filter(function (c) { return hiddenClasses.indexOf(c) < 0; }); }

  // A "don't want" class (collect.hidden_classes; case-insensitive).
  function isHiddenClass(label) {
    var l = String(label || '').trim().toLowerCase();
    return hiddenClasses.some(function (h) { return h.toLowerCase() === l; });
  }

  // A box of an unwanted class: of a hidden class, or rejected and of no class of the list (the
  // server's other_class rejects). Not drawn, listed or stepped through (display only; the file
  // keeps it). Other boxes of no listed class stay visible: a save would refuse them.
  function isUnwanted(b) {
    return isHiddenClass(b.label) || (b.state === 'rejected' && classes.indexOf(b.label) < 0);
  }

  // "Show boxes of": a box of another class is dimmed (display only; saving keeps every box).
  function isShown(b) {
    return !isUnwanted(b) && (!shownLabels.length || shownLabels.indexOf(b.label) >= 0);
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function round1(v) { return Math.round(v * 10) / 10; }

  function iou(a, b) {
    var x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]), x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
    var inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    var u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return u > 0 ? inter / u : 0;
  }

  var VEHICLES = ['car', 'truck', 'bus', 'van'];

  // Two labels that can name one object: the same class, or two vehicle classes.
  function sameObject(a, b) {
    return a === b || (VEHICLES.indexOf(a) >= 0 && VEHICLES.indexOf(b) >= 0);
  }

  // Stacked boxes: the boxes (not rejected, not unwanted) of one object, IoU >= 0.7 and a label
  // of the same object, grouped as NMS would. A group of 2 or more keeps one: a drawn box, else
  // an accepted one, else the highest score. {groups: [{members, keeper}], of: {index: group}}.
  function dupGroups() {
    var bs = boxes(), n = bs.length, parent = [], i, j;
    var live = function (b) { return b.state !== 'rejected' && !isUnwanted(b); };
    var root = function (x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (i = 0; i < n; i++) parent.push(i);
    for (i = 0; i < n; i++) {
      if (!live(bs[i])) continue;
      for (j = i + 1; j < n; j++) {
        if (live(bs[j]) && sameObject(bs[i].label, bs[j].label) && iou(bs[i].bbox, bs[j].bbox) >= 0.7) parent[root(j)] = root(i);
      }
    }
    var byRoot = {}, out = { groups: [], of: {} };
    for (i = 0; i < n; i++) {
      if (live(bs[i])) (byRoot[root(i)] = byRoot[root(i)] || []).push(i);
    }
    var rank = function (k) {
      var b = bs[k];
      return (b.state === 'added' ? 20 : (b.state === 'accepted' ? 10 : 0)) + (b.score != null ? b.score : 0);
    };
    Object.keys(byRoot).forEach(function (r) {
      var m = byRoot[r];
      if (m.length < 2) return;
      var g = { members: m, keeper: m.reduce(function (a, k) { return rank(k) > rank(a) ? k : a; }, m[0]) };
      out.groups.push(g);
      m.forEach(function (k) { out.of[k] = g; });
    });
    return out;
  }

  // The box of `i`'s stack it is compared with: the keeper, or (for the keeper) the most
  // overlapping other one.
  function dupPartner(i) {
    var g = dups.of[i];
    if (!g) return -1;
    if (g.keeper !== i) return g.keeper;
    var bs = boxes(), best = -1, bi = -1;
    g.members.forEach(function (k) {
      if (k === i) return;
      var v = iou(bs[i].bbox, bs[k].bbox);
      if (v > bi) { bi = v; best = k; }
    });
    return best;
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

  // `x`: zoom to the selected box and keep following the selection (Fit, the wheel or a pan stop it).
  function zoomToSelected() {
    var b = boxes()[selected];
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
    if (fitBtn) fitBtn.hidden = !zoomed;
    if (zoomBoxBtn) zoomBoxBtn.hidden = !boxes()[selected];
    if (zoomText) zoomText.textContent = zoomed ? Math.round(view.z * 100) + '%' : '';
    if (imgBtn) {
      imgBtn.textContent = imgLabel();
      imgBtn.title = 'View only: b cycles auto / off / bright / bright + equalised, [ and ] darker / brighter, Shift+0 resets';
    }
  }

  // ---- drawing ------------------------------------------------------------------------------

  function lineWidth() { return Math.max(2, Math.round(canvas.width / 400)); }

  function handlePoints(bb) {
    var cx = (bb[0] + bb[2]) / 2, cy = (bb[1] + bb[3]) / 2;
    return [['nw', bb[0], bb[1]], ['n', cx, bb[1]], ['ne', bb[2], bb[1]], ['e', bb[2], cy],
      ['se', bb[2], bb[3]], ['s', cx, bb[3]], ['sw', bb[0], bb[3]], ['w', bb[0], cy]];
  }

  function tagText(b, i) {
    var l = look(b);
    return (l === 'review' ? '? ' : '') + (b.difficult ? '?! ' : '') + (i + 1) + ' ' + b.label +
      (b.score != null ? ' ' + fmtScore(b.score) : '') +
      (b.source === 'auto' && b.triage === 'accept' && b.state !== 'rejected' ? ' ✓' : '');
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
    if (!cur || hideBoxes) return;
    var lw = lineWidth() / z;
    var font = Math.max(12, Math.round(W / 70)) / z;
    var pad = 4 / z;
    var th = font + pad;
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
      var dim = spot && !sel;
      ctx.save();
      var shown = isShown(b);
      ctx.globalAlpha = (l === 'rejected' ? 0.7 : 1) * (shown ? 1 : 0.2) * (dim ? 0.35 : 1);
      if (sel) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = lw * 3.5;
        ctx.strokeRect(x, y, w, h);
      }
      // Dashed: rejected or rejected on save; dotted: an unsure box.
      ctx.setLineDash(dashed ? [lw * 4, lw * 3] : (b.difficult ? [lw, lw * 1.5] : []));
      ctx.lineWidth = (sel ? 2 : 1) * (l === 'review' ? 1.5 : 1) * lw;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
      if (l === 'review') {
        // A dark inner line: the flagged boxes read as outlined.
        ctx.setLineDash([]);
        ctx.lineWidth = lw * 0.6;
        ctx.strokeStyle = '#000000';
        var inset = (sel ? 2.25 : 1.1) * lw;
        if (w > 2 * inset && h > 2 * inset) ctx.strokeRect(x + inset, y + inset, w - 2 * inset, h - 2 * inset);
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
        var text = (full || b.state === 'pending' || !b._checked) ? tagText(b, i) : String(i + 1);
        var tw = ctx.measureText(text).width + 2 * pad;
        // Above the box; at the frame top below it when there is room; else inside its top.
        var top;
        if (y >= th) top = y - th;
        else if (b.bbox[3] + th <= H) top = b.bbox[3];
        else top = y;
        var tx = clamp(x, 0, Math.max(0, W - tw));
        ctx.fillStyle = color;
        ctx.fillRect(tx, top, tw, th);
        ctx.fillStyle = '#000000';
        ctx.fillText(text, tx + pad, top + th - pad / 2);
      }
      ctx.restore();
    }

    // The selected box last (on top), the hovered one just before it.
    boxes().forEach(function (b, i) {
      if (!isUnwanted(b) && i !== selected && i !== hoverIdx) drawBox(b, i);
    });
    var hb = boxes()[hoverIdx];
    if (hb && hoverIdx !== selected && !isUnwanted(hb)) drawBox(hb, hoverIdx);
    if (sb && !isUnwanted(sb)) drawBox(sb, selected);
    if (sb && !(gesture && gesture.mode === 'draw' && gesture.moved)) {
      var hs = HANDLE_CSS * cssScale() / z;
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
  }

  function norm(d) {
    return [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
  }

  // The close-up: the selected box with a 40% margin, enlarged to the panel's width (a box under
  // 120 px at least 2x, at most 8x; the view then centres on the box), from the full-size image.
  function drawCrop() {
    if (!cropCanvas) return;
    var b = boxes()[selected];
    if (!b || !img || !img.complete || !img.naturalWidth) { cropWrap.hidden = true; return; }
    cropWrap.hidden = false;
    var W = canvas.width, H = canvas.height;
    var bw = Math.max(1, b.bbox[2] - b.bbox[0]), bh = Math.max(1, b.bbox[3] - b.bbox[1]);
    var aw = Math.max(120, cropWrap.clientWidth || 300);
    // Its height follows the window: the decision bar and the list stay in view under it.
    var ah = Math.round(Math.max(140, Math.min(wideSide ? 340 : 240, window.innerHeight * (wideSide ? 0.36 : 0.22))));
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
    c.setLineDash([]);
    cropTag(c, b, rx, ry, rh, pw, ph, dpr);
    cropCap.textContent = 'Box ' + (selected + 1) + ' (' + b.label + '), ' + Math.round(bw) + ' x ' + Math.round(bh) + ' px, ' +
      (Math.round(scale * 10) / 10) + 'x';
  }

  // The close-up's tag: the class at a glance (bold), the score, the decision so far (? flagged,
  // ✓ accepted / drawn, ✗ rejected, ?! unsure) and the labelers' class when they name another
  // one. Above the outline when there is room, else below it, else inside its top; kept within
  // the close-up.
  function cropTag(c, b, rx, ry, rh, pw, ph, dpr) {
    var l = look(b);
    var mark = b.difficult ? '?! ' : (l === 'review' ? '? ' : (l === 'accepted' || l === 'added' ? '✓ ' : (l === 'rejected' ? '✗ ' : '')));
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
        (b.agree === false ? ' (one labeler)' : ''));
    } else if (b.agree === false) parts.push('one labeler');
    if (b.vlm) {
      parts.push((b.vlm.model || 'VLM') + ': ' + (b.vlm.answer || 'nothing') +
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
    } else if (b.reason) parts.push(b.reason);
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

  // A box that is not selected: one row (number, class, state, score); a click selects it.
  function boxRow(b, i) {
    var li = el('li', 'compact st-' + b.state + ' lk-' + look(b));
    li.appendChild(el('span', 'swatch sw-' + look(b)));
    li.appendChild(el('span', 'bx-num', (isReview(b) ? '? ' : '') + String(i + 1)));
    li.appendChild(el('span', 'bx-name', b.label + (b.auto_label && b.auto_label !== b.label ? ' (' + b.auto_label + '?)' : '')));
    li.appendChild(shortState(b));
    if (b.state !== 'pending' && !b._checked) li.appendChild(el('span', 'rv-chip warn', 'to check'));
    if (dups.of[i] && dups.of[i].keeper !== i) li.appendChild(el('span', 'rv-chip warn', 'dup'));
    if (b.difficult) li.appendChild(el('span', 'rv-chip warn', 'unsure'));
    var rp = b.state === 'pending' ? repeatOf(b) : null;
    if (rp) li.appendChild(repeatChip(rp));
    li.appendChild(el('span', 'bx-score muted', fmtScore(b.score)));
    li.addEventListener('click', function () { select(i); });
    return li;
  }

  function boxItem(b, i) {
    if (i !== selected) return boxRow(b, i);
    var li = el('li', 'st-' + b.state + ' lk-' + look(b) + ' sel');
    var head = el('span', null);
    head.appendChild(el('span', 'swatch sw-' + look(b)));
    head.appendChild(document.createTextNode((isReview(b) ? '? ' : '') + String(i + 1) + ' '));
    li.appendChild(head);
    li.appendChild(classSelect(b, i));
    var chips = el('span', 'bx-chips');
    chips.appendChild(stateChip(b));
    if (b.state !== 'pending') chips.appendChild(el('span', 'rv-chip' + (b._checked ? ' ok' : ' warn'), b._checked ? 'checked' : 'to check'));
    // The reason of an `auto: review` is in the decision bar above: the chip alone here.
    var t = b.triage === 'review' ? el('span', 'rv-chip warn', 'auto: review') : triageChip(b);
    if (t && b.why && b.why.length) t.title = b.why.join('; ');
    if (t) chips.appendChild(t);
    if (dups.of[i] && dups.of[i].keeper !== i) chips.appendChild(el('span', 'rv-chip warn', 'dup'));
    if (b.difficult) chips.appendChild(el('span', 'rv-chip warn', 'unsure'));
    if (b.state === 'rejected' && b.reject_reason) chips.appendChild(el('span', 'rv-chip bad', reasonText(b.reject_reason)));
    li.appendChild(chips);
    var ev = evidence(b);
    if (ev) li.appendChild(el('span', 'bx-note', ev));
    // Accept / reject / keep / delete and "It's a <auto label>" are the decision bar's buttons.
    li.addEventListener('click', function () { select(i); });
    return li;
  }

  function renderSide() {
    boxList.textContent = '';
    var bs = boxes();
    dups = dupGroups();
    // Pending review boxes first; each keeps its number (the one on the canvas).
    var order = [];
    bs.forEach(function (b, i) { if (isReview(b)) order.push(i); });
    bs.forEach(function (b, i) { if (!isReview(b)) order.push(i); });
    var offScreen = 0, unwanted = 0;
    order.forEach(function (i) {
      if (isUnwanted(bs[i])) { unwanted++; return; }
      if (!isShown(bs[i])) { offScreen++; return; }
      boxList.appendChild(boxItem(bs[i], i));
    });
    if (offScreen) {
      boxList.appendChild(el('li', 'muted', offScreen + ' box(es) of other classes hidden by "Show boxes of" (still saved).'));
    }
    if (unwanted) {
      boxList.appendChild(el('li', 'muted', unwanted + ' box(es) of classes you do not want (hidden or not in the class list) not shown (kept in the file; pending proposals of them are rejected on save).'));
    }
    if (bs.length === unwanted && cur) boxList.appendChild(el('li', 'muted', 'No box. Drag on the image to draw one.'));
    emptyBox.checked = !!(cur && cur.labels.empty_confirmed);
    renderDecision();
  }

  // The frame's progress: the boxes the reviewer sees, undecided / decided but not checked / done.
  function progress() {
    var shown = boxes().filter(function (b) { return !isUnwanted(b) && isShown(b); });
    var u = shown.filter(function (b) { return b.state === 'pending'; }).length;
    var c = shown.filter(isUnchecked).length;
    return { total: shown.length, undecided: u, unchecked: c, done: shown.length - u - c };
  }

  function kbd(text) { return el('kbd', null, text); }

  // A decision button with its key; the frame keeps the focus (the bar is rebuilt).
  function decideButton(text, key, cls, onClick) {
    var b = button(text + ' ', function () { onClick(); canvas.focus({ preventScroll: true }); }, cls);
    if (key) b.appendChild(kbd(key));
    return b;
  }

  // The decision bar: the selected box's question, its answers, the frame's progress.
  function renderDecision() {
    decide.textContent = '';
    if (!cur) return;
    var p = progress();
    var b = boxes()[selected];
    if (!b) {
      decide.appendChild(el('div', 'rv-q', p.total
        ? (p.undecided || p.unchecked ? 'No box selected: j selects the next one, drag on the frame to draw one.'
          : (missingAsked === cur.frame ? 'Anything missing? Look over the whole frame and drag to draw a missed object; s saves.' : 'Every box decided: s saves.'))
        : 'No box: drag on the frame to draw one, or e confirms it empty.'));
    } else {
      var i = selected;
      var q = el('div', 'rv-q');
      q.appendChild(el('span', 'rv-q-num', 'Box ' + (i + 1) + ' of ' + boxes().length + ' · '));
      var verb, yes, no;
      if (b.source === 'human') { verb = 'Drawn: '; yes = 'Keep'; no = 'Delete'; }
      else if (b.state === 'accepted') { verb = 'Accepted as '; yes = 'Right, keep'; no = 'No, reject'; }
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
      if (b.difficult) why.unshift('marked unsure: kept, not trained on (YOLO), ignored by the benchmark');
      if (b.state === 'rejected' && b.reject_reason) why.unshift('why: ' + reasonText(b.reject_reason));
      if (why.length) decide.appendChild(el('div', 'rv-q-why muted', why.join(' · ')));
      var rp = b.state === 'pending' ? repeatOf(b) : null;
      var btns = el('div', 'rv-q-btns');
      btns.appendChild(decideButton(yes, 'a', 'ok' + (rp && rp.verdict === 'accepted' ? ' suggest' : ''), acceptSelected));
      btns.appendChild(decideButton(no, 'r', 'bad' + (rp && rp.verdict === 'rejected' ? ' suggest' : ''), rejectSelected));
      if (b.auto_label && b.auto_label !== b.label) {
        var al = b.auto_label;
        btns.appendChild(decideButton('It\'s a ' + al, null, 'ok', function () { relabelNext(i, al); }));
      }
      var other = decideButton('It\'s a…', '1-9', 'secondary', function () { classGrid = !classGrid; renderDecision(); });
      other.setAttribute('aria-expanded', classGrid ? 'true' : 'false');
      btns.appendChild(other);
      var unsure = decideButton(b.difficult ? 'Not unsure' : 'Unsure', 'd', 'secondary', toggleUnsure);
      unsure.title = 'A real object you cannot name for sure: kept, left out of YOLO training, ignored by the benchmark set';
      btns.appendChild(unsure);
      decide.appendChild(btns);
      var mate = dupPartner(i);
      if (mate >= 0) {
        var mb = boxes()[mate];
        var dl = el('div', 'rv-q-dup');
        dl.appendChild(el('span', null, 'Overlaps box ' + (mate + 1) + ' (' + mb.label + (mb.score != null ? ' ' + fmtScore(mb.score) : '') + ')' +
          (dups.of[i].keeper === i ? ', this one kept as the best' : '')));
        dl.appendChild(decideButton('Reject as duplicate', 'u', 'bad', function () { rejectWith(i, 'duplicate'); }));
        decide.appendChild(dl);
      }
      // Repeats: what the reviewer decided at this spot before, and the spot on every frame.
      if (b.source !== 'human') {
        var sl = el('div', 'rv-q-spot');
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
      cap.title = 'Optional: why box ' + (ri + 1) + ' is wrong (Shift+1-9)';
      rr.appendChild(cap);
      REASONS.forEach(function (r, n) {
        var cb = button('', function () { setReason(n); canvas.focus({ preventScroll: true }); },
          'secondary' + (rb.reject_reason === r[0] ? ' active' : ''));
        cb.title = r[1] + ' (Shift+' + (n + 1) + ')';
        cb.appendChild(kbd(String(n + 1)));
        cb.appendChild(document.createTextNode(r[2]));
        rr.appendChild(cb);
      });
      decide.appendChild(rr);
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
    if (p.total || stackBtn || repeatBtn) {
      var line = el('div', 'rv-progress');
      if (p.total) {
        var bar = el('progress');
        bar.max = p.total;
        bar.value = p.done;
        line.appendChild(bar);
      }
      var row = el('div', 'rv-progress-row');
      row.appendChild(el('span', 'muted', p.total ? p.done + '/' + p.total + ' done · ' + p.undecided + ' undecided' + (p.unchecked ? ' · ' + p.unchecked + ' to check' : '') : ''));
      if (stackBtn) row.appendChild(stackBtn);
      if (repeatBtn) row.appendChild(repeatBtn);
      line.appendChild(row);
      decide.appendChild(line);
    }
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
    // The page too when the decision bar is off screen (one column: the panel does not stick).
    if (page) {
      var dr = decide.getBoundingClientRect();
      if (dr.top < 0 || dr.bottom > window.innerHeight) decide.scrollIntoView({ block: 'nearest' });
    }
  }

  // A new frame: the page scrolls as little as needed to show the whole frame and the side panel
  // (with its Save row; from the grid's top when they are taller than the window), the box list
  // starts at its top.
  function revealFrame() {
    boxList.scrollTop = 0;
    revealSelected();
    if (!grid) return;
    var top = grid.getBoundingClientRect().top;
    var bottom = Math.max(canvas.getBoundingClientRect().bottom, side ? side.getBoundingClientRect().bottom : 0);
    var by = 0;
    if (top < 0) by = top - 8;
    else if (bottom > window.innerHeight) by = Math.min(top - 8, bottom - window.innerHeight + 8);
    if (by) window.scrollBy(0, by);
  }

  // ---- edits --------------------------------------------------------------------------------

  function select(i, page) {
    selected = (i >= 0 && i < boxes().length) ? i : -1;
    if (view.follow && boxes()[selected]) zoomTo(boxes()[selected]);
    else panTo(boxes()[selected]);
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
      else { b.state = state; b._checked = true; delete b.reject_reason; }
      if (state !== 'rejected' && cur.labels.empty_confirmed) cur.labels.empty_confirmed = false;
    }
    var next = nextUndecided(i);
    select(next >= 0 ? next : i, true);
    sayProgress();
  }

  // Every rejection by the reviewer goes through here (no undo step, no move: the callers).
  // `why`: the box's `reject_reason` ('duplicate', ...); none keeps an earlier one. A rejected
  // box is not "unsure".
  function markRejected(b, why) {
    b.state = 'rejected';
    b._checked = true;
    delete b.difficult;
    if (why) b.reject_reason = why;
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
  var reasonAsk = null;   // {frame, i, timer}: the chips offered after a rejection

  function reasonText(r) {
    for (var n = 0; n < REASONS.length; n++) if (REASONS[n][0] === r) return REASONS[n][1];
    return r;
  }

  function offerReason(i) {
    clearReason(true);
    reasonAsk = { frame: cur.frame, i: i, timer: setTimeout(function () { clearReason(); }, REASON_MS) };
    renderDecision();
  }

  function clearReason(quiet) {
    if (!reasonAsk) return;
    clearTimeout(reasonAsk.timer);
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
    if (b.reject_reason === r[0]) delete b.reject_reason; else b.reject_reason = r[0];
    say('Box ' + (i + 1) + ' rejected' + (b.reject_reason ? ': ' + r[1] : '') + '.');
    clearReason(true);
    refresh();
  }

  // `d`: the selected box is a real object the reviewer cannot name for sure (VOC "difficult"):
  // kept, left out of the YOLO export, ignored by the benchmark set; then the next box. Again on
  // an unsure box: no longer unsure (its decision stays).
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
    b.state = b.source === 'human' ? 'added' : 'accepted';
    b._checked = true;
    if (cur.labels.empty_confirmed) cur.labels.empty_confirmed = false;
    var next = nextUndecided(selected);
    select(next >= 0 ? next : selected, true);
    sayProgress();
  }

  // The non-drawn boxes of the stacks that are not their keeper.
  function dupDrops(d) {
    var out = [];
    d.groups.forEach(function (g) {
      g.members.forEach(function (k) { if (k !== g.keeper && boxes()[k].source !== 'human') out.push(k); });
    });
    return out;
  }

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

  // A decision of the reviewer's own (the server's rule, repeats.rs `counts`).
  function lookedAt(b) {
    return b.source !== 'human' && (b.state === 'accepted' || b.state === 'rejected') &&
      ['low_score', 'other_class', 'repeat_spot'].indexOf(b.reason) < 0;
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
    b._checked = true;
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

  // `a`: accept the pending non-human boxes saving would accept anyway; hints stay pending.
  function acceptProposals() {
    if (!cur) return;
    var todo = boxes().filter(function (b) {
      return b.state === 'pending' && b.source !== 'human' && !isUnwanted(b) && resolved(b) === 'accepted';
    });
    var hints = boxes().filter(function (b) { return isHint(b) && !isUnwanted(b); }).length;
    if (!todo.length) {
      say(hints ? hints + ' hint box(es) stay pending (rejected on save unless you accept them); no other proposal.' : 'No pending proposal.');
      return;
    }
    snapshot();
    todo.forEach(function (b) { b.state = 'accepted'; });
    cur.labels.empty_confirmed = false;
    say('Accepted ' + todo.length + ' box(es)' + (hints ? '; ' + hints + ' hint box(es) stay pending (rejected on save unless you accept them).' : '.'));
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
    var r = (HANDLE_CSS + 2) * cssScale() / view.z;
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

  canvas.addEventListener('pointerdown', function (ev) {
    if (!cur) return;
    canvas.focus();
    if (ev.button === 1 || ev.button === 2 || (ev.button === 0 && spaceDown)) {
      ev.preventDefault();
      gesture = { mode: 'pan', cx: ev.clientX, cy: ev.clientY, ox: view.ox, oy: view.oy, moved: false };
      view.follow = false;
      canvas.setPointerCapture(ev.pointerId);
      setCursor('grabbing');
      return;
    }
    if (ev.button !== 0) return;
    var p = framePoint(ev);
    var g = { mode: 'draw', cx: ev.clientX, cy: ev.clientY, x0: p[0], y0: p[1], x1: p[0], y1: p[1], moved: false };
    if (!hideBoxes && selected >= 0) {
      var h = handleAt(p);
      if (h) { g.mode = 'resize'; g.handle = h; }
      else if (insideSelected(p)) g.mode = 'move';
      if (g.mode !== 'draw') { g.idx = selected; g.orig = boxes()[selected].bbox.slice(); }
    }
    gesture = g;
    canvas.setPointerCapture(ev.pointerId);
  });

  canvas.addEventListener('pointermove', function (ev) {
    lastPointer = ev;
    var g = gesture;
    if (!g) { hoverCursor(ev); return; }
    if (!g.moved) {
      if (Math.abs(ev.clientX - g.cx) > CLICK_CSS || Math.abs(ev.clientY - g.cy) > CLICK_CSS) {
        g.moved = true;
        if ((g.mode === 'move' || g.mode === 'resize') && boxes()[g.idx]) {
          snapshot();
          markEdited(boxes()[g.idx]);
        }
      }
      if (!g.moved) return;
    }
    if (g.mode === 'pan') {
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
    var g = gesture;
    if (!g) return;
    gesture = null;
    if (g.mode === 'pan') { hoverCursor(ev); draw(); return; }
    if (g.moved) {
      if (g.mode === 'draw') {
        var r = norm(g);
        if (r[2] - r[0] >= MIN_BOX && r[3] - r[1] >= MIN_BOX) {
          snapshot();
          boxes().push({
            label: classSel.value || classes[0] || 'object',
            bbox: r.map(round1),
            source: 'human', state: 'added', _checked: true
          });
          cur.labels.empty_confirmed = false;
          selected = boxes().length - 1;
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
    // A click: the smallest box under the point.
    select(boxAt([g.x0, g.y0]));
    hoverCursor(ev);
  });

  canvas.addEventListener('pointercancel', function () {
    var g = gesture;
    gesture = null;
    // A cancelled move / resize keeps what it did (it is one undo step).
    if (g && g.moved && g.mode !== 'draw' && g.mode !== 'pan') refresh(); else draw();
  });
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
  if (fitBtn) fitBtn.addEventListener('click', function () { fit(); canvas.focus(); });
  if (zoomBoxBtn) zoomBoxBtn.addEventListener('click', function () { zoomToSelected(); canvas.focus(); });

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
    resizeTimer = setTimeout(function () { resizeTimer = null; draw(); }, 100);
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
    };
    add('Reviewed', c.reviewed || 0);
    add('Remaining', c.remaining || 0);
    add('Skipped', c.skipped || 0);
    add('Confirmed empty', c.empty_confirmed || 0);
    add('Rejected boxes', c.rejected_boxes || 0);
    if (c.auto_review != null) add('Auto: to review', c.auto_review);
    if (c.auto_accept != null) add('Auto: accept-ready', c.auto_accept);
    if (c.repeats_only) add('Repeat-only', c.repeats_only);
    var per = c.per_class || {};
    var keys = Object.keys(per);
    add('Per class', keys.length ? keys.map(function (k) { return k + ' ' + per[k]; }).join(', ') : 'none yet');
    add('Matching', j.position ? (j.position + ' of ' + j.total) : (j.total || 0));
    if (j.unreadable) add('Unreadable', j.unreadable);
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

  // The Class filter: remember the classes the queue saw.
  function fillLabels(list) {
    (list || []).forEach(function (c) { if (seenLabels.indexOf(c) < 0) seenLabels.push(c); });
    if (labelPicker) labelPicker.setEntries(buildEntries());
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
  }

  function showNone(on) {
    $('rv-none').hidden = !on;
    canvas.hidden = on;
    if (on && cropWrap) cropWrap.hidden = true;
  }

  function prefetch() {
    var next = upcoming[0];
    if (!next || (cur && next === cur.frame)) return;
    prefetchImg = new Image();
    prefetchImg.src = '/dataset/image?path=' + encodeURIComponent(next);
  }

  // `nav`: 'back' (`p`) or 'forward' (`n` after Back); else a move on, which forgets the frames
  // Back left.
  function loadFrame(frame, nav) {
    clearReason(true);
    return api('/v1/review/labels?frame=' + encodeURIComponent(frame)).then(function (j) {
      if (cur && cur.frame !== j.frame) {
        if (nav === 'back') fwdStack.push(cur.frame);
        else { backStack.push(cur.frame); if (nav !== 'forward') fwdStack = []; }
        if (backStack.length > 200) backStack.shift();
      }
      cur = j;
      loadHistory(camOf(j.frame));
      if (Array.isArray(j.classes)) classes = j.classes;
      if (Array.isArray(j.hidden)) hiddenClasses = j.hidden;
      undo = []; dirty = false; guard = null; lastNudge = null; gesture = null;
      emptyBackup = null; missingAsked = null; frameStart = Date.now();
      canvas.width = j.width || 640;
      canvas.height = j.height || 360;
      view.z = 1; view.ox = 0; view.oy = 0;
      hoverIdx = -1;
      selected = nextUndecided(-1);
      // Following (`x`): the new frame opens zoomed on its first box.
      if (view.follow && boxes()[selected]) zoomTo(boxes()[selected]);
      img = new Image();
      img.onload = draw;
      img.src = j.image_url;
      showNone(false);
      setMeta();
      refresh();
      revealFrame();
      prefetch();
    });
  }

  // The next frame after `after` (the first without), `note` said first; `countsOnly` refreshes
  // the counts and lists and keeps the frame on screen (and its edits).
  function loadNext(after, note, countsOnly) {
    var p = filterParams();
    if (after) p.set('after', after);
    return api('/v1/review/next?' + p.toString()).then(function (j) {
      showCounts(j);
      fillCameras(j.cameras);
      fillLabels(j.labels);
      if (countsOnly) return;
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
    }).catch(function (e) { say(e.message, true); });
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
      var n = reviewCount();
      if (n > 0 && !(guard && guard.frame === cur.frame && guard.count === n)) {
        guard = { frame: cur.frame, count: n };
        say(n + ' box(es) the auto-labeler flagged still need a decision (marked ?): accept or reject them, or save again to use the defaults.', true);
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
        say('Anything missing? Look over the whole frame and drag to draw a missed object; s saves.');
        return;
      }
    }
    busy = true;
    var body = JSON.parse(JSON.stringify(cur.labels));   // keeps auto_run, bulk_accepted, ...
    // Proposals of a "don't want" class are not shown: rejected rather than resolved by default.
    body.boxes.forEach(function (b) {
      delete b._checked;
      if (b.state === 'pending' && b.source !== 'human' && isHiddenClass(b.label)) {
        b.state = 'rejected';
        b.reason = 'other_class';
      }
    });
    body.version = body.version || 1;
    body.status = status;
    body.empty_confirmed = !!cur.labels.empty_confirmed;
    body.classes_version = cur.labels.classes_version || '';
    var frame = cur.frame, w = cur.width, h = cur.height;
    var url = '/v1/review/labels?frame=' + encodeURIComponent(frame);
    if (cur.revision != null) url += '&revision=' + encodeURIComponent(cur.revision);
    postJson(url, body).then(function (j) {
      dirty = false;
      if (status === 'reviewed') { noteSaved(body); noteHistory(frame, body, w, h); }
      // After Back: on to the frame Back left (the counts refresh), else the queue's next.
      if (fwdStack.length) {
        say(j.message || 'Saved.');
        loadNext(null, null, true);
        return loadFrame(fwdStack.pop(), 'forward');
      }
      return loadNext(frame, j.message || 'Saved.');
    }).catch(function (e) {
      if (e.conflict) showConflict(e.message);
      else say(e.message, true);
    }).then(function () { busy = false; });
  }

  // `n`: the next frame without saving; after Back, the frame Back left.
  function skipToNext() {
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
      if (b.source === 'human') return;
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
  if (imgBtn) imgBtn.addEventListener('click', function () { cycleImg(); canvas.focus({ preventScroll: true }); });
  if (missingBox) {
    missingBox.checked = askMissing;
    missingBox.addEventListener('change', function () { askMissing = missingBox.checked; setPref('askMissing', askMissing); });
  }
  emptyBox.addEventListener('change', function () { confirmEmpty(emptyBox.checked); });
  // A filter change with unsaved edits asks first; kept, the frame stays and the counts follow.
  function filterReload() {
    if (leaveOk()) { loadNext(); return; }
    loadNext(null, null, true);
    say('Kept this frame with its unsaved changes; the new filters apply from the next frame.');
  }
  var filterTimer = null;
  function filterChanged() {
    if (filterTimer) clearTimeout(filterTimer);
    filterTimer = setTimeout(function () { filterTimer = null; filterReload(); }, 200);
  }
  filtersForm.addEventListener('change', function (ev) {
    // The class picker reports through its own onChange (one reload per change).
    if (ev.target && ev.target.closest && ev.target.closest('.cp')) return;
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
    ['Decide', '1-9, 0', null],
    ['Decide', 'u', 'reject the selected box as a duplicate of the box it overlaps'],
    ['Decide', 'Shift+U', 'stacked boxes: keep the best of each stack, reject the others as duplicates'],
    ['Decide', 'd', 'unsure: a real object you cannot name for sure (kept, left out of YOLO training, ignored by the benchmark set; ?! on the box); again: not unsure'],
    ['Decide', 'Shift+1-9', 'after a rejection, or on a rejected box: why it was wrong (optional; nothing there, insect / web, rain / snow, IR glare, reflection, part of the scene, other object, too small, duplicate)'],
    ['Decide', 'Shift+A', 'accept proposals: the pending boxes saving would accept (hints stay pending)'],
    ['Decide', 'e', 'confirmed empty (rejects every box)'],
    ['Decide', 'Delete', 'delete a drawn box, reject a model box'],
    ['Decide', 'z', 'undo'],
    ['Decide', 's, Enter', 'save & next; asks first about flagged (?) boxes, boxes not checked yet, then "Anything missing?" (Panel options)'],
    ['Decide', 'drag', 'on a handle: resize; inside the selected box: move; elsewhere: draw a box'],
    ['Decide', 'arrows', 'move the selected box 1 px (Shift: 10 px)'],
    ['Decide', '', 'The next box is the next flagged (?) one, then the next pending one, then each accepted / rejected box not checked yet (check pass, marked to check).'],
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
    if (t && (t.tagName === 'BUTTON' || t.tagName === 'SUMMARY' || t.tagName === 'INPUT') && (k === 'Enter' || k === ' ')) return;
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
    } else if (k === 'l') { showTags = !showTags; draw(); }
    else if (k === 'v') {
      spotlight = !spotlight;
      setPref('spotlight', spotlight);
      say(spotlight ? 'Spotlight on: the selected box stands out.' : 'Spotlight off: every box at full strength.');
      draw();
    } else if (k === 'x') zoomToSelected();
    else if (k === 'f') fit();
    else if (k === 'b') cycleImg();
    else if (k === '[' || k === ']') stepGamma(k === ']' ? 1 : -1);
    else if (ev.code === 'Digit0' && ev.shiftKey) setImgView('auto', 1);
    else if (k === 'p') goBack();
    else if (k === 'd') toggleUnsure();
    else if (k === 'u') { if (ev.shiftKey) rejectDuplicates(); else rejectDupSelected(); }
    else if (k === '?') openKeys();
    else if (k === 'ArrowLeft' || k === 'ArrowRight' || k === 'ArrowUp' || k === 'ArrowDown') {
      var step = ev.shiftKey ? 10 : 1;
      handled = selected >= 0 && nudge(
        k === 'ArrowLeft' ? -step : (k === 'ArrowRight' ? step : 0),
        k === 'ArrowUp' ? -step : (k === 'ArrowDown' ? step : 0));
    } else if (k === 'a') {
      if (ev.shiftKey) acceptProposals(); else acceptSelected();
    } else if (k === 'r') rejectSelected();
    else if (k === 'n') skipToNext();
    else if (k === 'e') confirmEmpty(!(cur && cur.labels.empty_confirmed));
    else if (k === 'z') doUndo();
    else if (k === 's' || k === 'Enter') save('reviewed');
    else if (k === 'j') select(stepShown(1), true);
    else if (k === 'k') select(stepShown(-1), true);
    else if (k === 'Delete' || k === 'Backspace') {
      if (selected >= 0) {
        if (boxes()[selected].source === 'human') removeBox(selected);
        else rejectWith(selected, null);
      }
    } else if (k === 'Escape') select(-1);
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

  // ---- auto-label: queue a job, follow it, bulk accept -------------------------------------

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

  var alForm = $('rv-autolabel');
  var alStatus = $('rv-al-status');
  var alProgress = $('rv-al-progress');
  var polling = null;   // {id, timer}

  // Label with: labelers (hidden fields, as before) or a model (model-picker.js: model, device,
  // threshold, pre-accept score, the licence / large-download confirms). The blocks of the other
  // mode are hidden and disabled, so they are not posted.
  var MP = window.ModelPicker;
  var alModelPicker = null;
  function alMode() {
    var r = alForm && alForm.querySelector('input[name="mode"]:checked');
    return r ? r.value : 'labelers';
  }
  function applyAlMode() {
    if (!alForm) return;
    var m = alMode();
    Array.prototype.forEach.call(alForm.querySelectorAll('[data-al-mode]'), function (blk) {
      var off = blk.getAttribute('data-al-mode') !== m;
      blk.hidden = off;
      Array.prototype.forEach.call(blk.querySelectorAll('input, select, textarea'), function (f) { f.disabled = off; });
    });
  }
  if (alForm) {
    var mpHolder = $('rv-al-model-picker');
    if (mpHolder && MP) {
      alModelPicker = MP.create(mpHolder, {
        onLoad: function (j) {
          var t = alForm.elements.threshold;
          if (t && j.default_threshold && t.value === t.defaultValue) t.value = j.default_threshold;
        }
      });
    }
    Array.prototype.forEach.call(alForm.querySelectorAll('input[name="mode"]'), function (r) {
      r.addEventListener('change', applyAlMode);
    });
    applyAlMode();
  }

  // The filters of the queue become the job's frame filter.
  function autolabelFrames() {
    var parts = [];
    // `label` (a comma list) goes in the form's own `label` field, not in this comma list.
    Array.prototype.forEach.call(filtersForm.elements, function (e) {
      if (e.name && e.name !== 'label' && e.value) parts.push(e.name + '=' + e.value);
    });
    var lim = alForm.elements.limit && alForm.elements.limit.value;
    if (lim) parts.push('limit=' + lim);
    var again = $('rv-al-again');
    if (again && again.checked) parts.push('again=1');
    return parts.join(',');
  }

  function describeTask(t) {
    if (t.state === 'queued') return 'Queued: ' + (t.title || 'auto-label') + ' (waiting for the queue).';
    var s = t.stage ? String(t.stage).replace(/_/g, ' ') : 'starting';
    return (t.title ? t.title + ': ' : '') + s + (t.steps ? ' (' + t.step + '/' + t.steps + ')' : '') +
      (t.percent != null ? ', ' + t.percent + '%' : '') + (t.detail ? ': ' + t.detail : '');
  }

  function stopPolling() {
    if (polling) clearInterval(polling.timer);
    polling = null;
  }

  function pollTask(id) {
    stopPolling();
    polling = { id: id, timer: setInterval(function () { tick(id); }, 3000) };
    tick(id);
  }

  function tick(id) {
    fetch('/v1/training', { headers: { Accept: 'application/json' } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data || !polling || polling.id !== id) return;
        var t = (data.tasks || []).filter(function (x) { return x.id === id; })[0];
        if (!t) return;
        if (t.state === 'queued' || t.state === 'running') {
          report(alStatus, [describeTask(t)]);
          alProgress.hidden = false;
          alProgress.value = t.percent || 0;
          return;
        }
        stopPolling();
        alProgress.hidden = true;
        if (t.state === 'done') report(alStatus, ['Done: ' + (t.summary || t.title || 'auto-label finished')]);
        else if (t.state === 'failed') report(alStatus, ['Failed: ' + (t.error || 'see the Training page.')]);
        else report(alStatus, ['Cancelled.']);
        alStatus.className = 'review-report ' + (t.state === 'done' ? 'ok-text' : 'error-text');
        reloadQueue();
      })
      .catch(function () {});
  }

  if (alForm) {
    alForm.addEventListener('submit', function (ev) {
      alForm.elements.frames.value = autolabelFrames();
      if (alForm.elements.label) alForm.elements.label.value = filtersForm.elements.label ? filtersForm.elements.label.value : '';
      if (!window.fetch || !window.URLSearchParams) return;   // the plain form post
      ev.preventDefault();
      var picked = alPicker ? alPicker.getSelected() : [];
      if (alPicker && !picked.length) {
        alStatus.className = 'review-report error-text';
        report(alStatus, ['Choose at least one class to look for.']);
        return;
      }
      if (alMode() === 'model') {
        var problem = !alModelPicker ? 'The model list is not available (reload the page).' :
          (alModelPicker.problem() ||
            MP.util.acceptProblem(alForm.elements.threshold.value, alForm.elements.model_accept_score.value));
        if (problem) {
          alStatus.className = 'review-report error-text';
          report(alStatus, [problem]);
          return;
        }
      }
      var body = new URLSearchParams();
      Array.prototype.forEach.call(alForm.elements, function (e) {
        if (!e.name || e.disabled) return;
        if ((e.type === 'checkbox' || e.type === 'radio') && !e.checked) return;
        body.append(e.name, e.value);
      });
      picked.forEach(function (c) { body.append('classes', c); });
      alStatus.className = 'review-report';
      // Listed, wanted classes left unticked: offer to stop wanting them altogether.
      var skipped = alPicker ? visible().filter(function (c) { return picked.indexOf(c) < 0; }) : [];
      var hideFirst = skipped.length > 0 && window.confirm('Hide and auto-reject ' + skipped.join(', ') +
        ' in Review? (Cancel = only skip them in this run)');
      report(alStatus, [hideFirst ? 'Hiding ' + skipped.join(', ') + '...' : 'Queueing...']);
      var hidden_ = !hideFirst ? Promise.resolve() :
        postJson('/v1/review/classes', { classes: classes.slice(), hidden: hiddenClasses.concat(skipped) })
          .then(function (j) {
            afterClassesSaved(j);
            var lines = [j.message || 'Saved.'];
            var auto = unwantedNote(j);
            if (auto) lines.push(auto);
            report(alStatus, lines);
            return askReviewedUnwanted(j.reviewed_with_unwanted, alStatus);
          });
      var ready = hidden_.then(function () {
        return alPicker && catalog ? CP.ensureInList(picked, catalog) : catalog;
      });
      ready.then(function () {
        if (alPicker) { refreshPickers(); }
        return api('/training/autolabel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json' },
          body: body.toString()
        });
      }).then(function (j) {
        var m = j.message || 'Queued.';
        var q = /^Queued/.test(m) ? m : 'Queued: ' + m;
        // After hiding classes: their report stays above.
        if (hideFirst) alStatus.appendChild(el('div', null, q));
        else report(alStatus, [q]);
        if (j.id) pollTask(j.id);
      }).catch(function (e) {
        alStatus.className = 'review-report error-text';
        report(alStatus, [e.cancelled ? 'Not queued: the class list was left as it is.' : e.message]);
      });
    });
  }

  // A job queued earlier (this page or the Training page): follow it.
  if (alStatus) {
    fetch('/v1/training', { headers: { Accept: 'application/json' } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data || polling) return;
        var busyTasks = (data.tasks || []).filter(function (t) {
          return t.kind === 'autolabel' && (t.state === 'queued' || t.state === 'running');
        });
        var t = busyTasks.filter(function (x) { return x.state === 'running'; })[0] || busyTasks[0];
        if (t) pollTask(t.id);
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

  // What a class save rejected on its own (`auto_rejected` of POST /v1/review/classes), or ''.
  function unwantedNote(j) {
    var a = j && j.auto_rejected;
    if (!a || !(a.boxes || 0)) return '';
    var rejected = a.boxes - (a.relabelled || 0);
    var text = 'Rejected ' + rejected + ' pending box(es) of unwanted classes (newly: ' +
      (j.newly_unwanted || []).join(', ') + ') on ' + (a.frames || 0) + ' unreviewed frame(s) (reason other_class).';
    if (a.relabelled) text += ' Renamed ' + a.relabelled + ' box(es) to the class list\'s spelling.';
    if (a.failed && a.failed.length) text += ' ' + a.failed.length + ' frame(s) not changed: ' + a.failed.join('; ');
    return text;
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

  // ---- export / import / classes ------------------------------------------------------------

  function list(text) {
    return String(text || '').split(/[,\n]/).map(function (s) { return s.trim(); }).filter(Boolean);
  }

  $('rv-export').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var f = ev.target;
    var box = $('rv-export-msg');
    var formats = Array.prototype.filter.call(f.querySelectorAll('input[name=fmt]'), function (c) { return c.checked; })
      .map(function (c) { return c.value; });
    var body = {
      formats: formats,
      holdout_cameras: list(f.elements.holdout_cameras.value),
      holdout_days: list(f.elements.holdout_days.value),
      holdout_fraction: Number(f.elements.holdout_fraction.value || 0.2),
      benchmark_split: f.elements.benchmark_split.value,
      include_unreviewed: f.elements.include_unreviewed.checked,
      labels: list(f.elements.labels.value),
      include_empty: f.elements.include_empty.checked
    };
    if (f.elements.out_dir.value.trim()) body.out_dir = f.elements.out_dir.value.trim();
    report(box, ['Exporting...']);
    postJson('/v1/review/export', body).then(function (j) {
      var r = j.report;
      var per = Object.keys(r.per_class || {}).map(function (k) { return k + ' ' + r.per_class[k]; }).join(', ');
      var lines = [
        'Exported ' + r.frames + ' frame(s) (' + r.train + ' train, ' + r.val + ' held out, ' +
          r.empty_confirmed + ' confirmed empty) to ' + r.out_dir,
        'Boxes per class: ' + (per || 'none') + '; ' + r.hard_negative_boxes + ' rejected box(es) in hard_negatives.json'
      ];
      var cf = r.class_filter || {};
      if ((cf.labels && cf.labels.length) || cf.include_empty === false) {
        lines.push('Class filter: ' + ((cf.labels && cf.labels.length) ? cf.labels.join(', ') : 'every class') +
          (cf.include_empty === false ? ', without confirmed-empty frames' : ', with confirmed-empty frames') +
          '; ' + (r.filtered_out || 0) + ' reviewed frame(s) left out');
      }
      if (r.unreviewed_frames) lines.push(r.unreviewed_frames + ' unreviewed frame(s) added to the X-AnyLabeling folder');
      if (r.benchmark_dataset) lines.push('Benchmark: blue-onyx-prism benchmark --dataset "' + r.benchmark_dataset + '"');
      report(box, lines, r.warnings);
    }).catch(function (e) { report(box, [e.message]); });
  });

  $('rv-import').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var f = ev.target;
    var box = $('rv-import-msg');
    report(box, ['Importing...']);
    postJson('/v1/review/import', {
      dir: f.elements.dir.value.trim(),
      treat_unchecked_as_reviewed: f.elements.treat_unchecked_as_reviewed.checked
    }).then(function (j) {
      var r = j.report;
      report(box, ['Imported ' + r.imported + ' frame(s); ' + r.unchecked + ' unchecked file(s) left alone; ' +
        r.unmatched + ' file(s) name no frame of the dataset.'], (r.errors || []).concat(r.warnings || []));
      loadNext();
    }).catch(function (e) { report(box, [e.message]); });
  });

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

  // The Classes card. Without the picker the comma text field saves as before.
  function afterClassesSaved(j) {
    classes = j.classes;
    if (j.hidden) hiddenClasses = j.hidden;
    if (catalog) { catalog.classes = classes.slice(); catalog.hidden = hiddenClasses.slice(); }
    hideSet = {};
    classes.forEach(function (c) { hideSet[c] = hiddenClasses.indexOf(c) >= 0; });
    fillClassSelect();
    fillKeyHint();
    refreshPickers();
    refresh();
  }

  $('rv-classes').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var box = $('rv-classes-msg');
    var body;
    if (classesPicker) {
      var list_ = classesPicker.getSelected();
      if (!list_.length) {
        box.textContent = 'Keep at least one class in the list.';
        box.className = 'form-msg error-text';
        return;
      }
      var change = CU.classChange(classes, list_);
      if (change.idsChange && !window.confirm(
        (change.removed.length ? 'Remove ' + change.removed.join(', ') + ' from the class list. ' : '') +
        (change.reordered ? 'The order of the classes changes. ' : '') +
        'YOLO class ids of later exports change (frames you exported before keep the old ids). Pending boxes of a ' +
        'removed class are rejected on the unreviewed frames; for reviewed frames you are asked next. Continue?')) {
        box.textContent = 'Not saved.';
        box.className = 'form-msg';
        return;
      }
      body = { classes: list_, hidden: hideChecked(list_) };
    } else {
      body = { classes: list(ev.target.elements.classes.value) };
    }
    postJson('/v1/review/classes', body).then(function (j) {
      afterClassesSaved(j);
      var lines = [j.message || 'Saved.'];
      var auto = unwantedNote(j);
      if (auto) lines.push(auto);
      box.className = 'form-msg ok-text';
      report(box, lines);
      return askReviewedUnwanted(j.reviewed_with_unwanted, box).then(function () {
        if (j.auto_rejected && j.auto_rejected.boxes) return reloadQueue();
      });
    }).catch(function (e) { box.textContent = e.message; box.className = 'form-msg error-text'; });
  });

  // ---- class pickers ------------------------------------------------------------------------

  var hideSet = {};   // Classes card: class -> "don't want" ticked

  function hideChecked(selectedList) {
    return selectedList.filter(function (c) { return hideSet[c]; });
  }

  function refreshPickers() {
    var entries = buildEntries();
    if (labelPicker) labelPicker.setEntries(entries);
    if (showPicker) showPicker.setEntries(entries);
    if (alPicker) {
      alPicker.setEntries(entries);
      if (catalog) alPicker.setQuickSets(CU.quickSetsFor(catalog));
    }
    if (classesPicker) classesPicker.setEntries(entries);
  }

  function initPickers() {
    if (!CP) return;
    var entries = buildEntries();
    var labelHidden = $('rv-label');
    var fb = $('rv-label-fallback');
    var holder = $('rv-label-picker');
    if (holder && labelHidden) {
      var reload = filterChanged;
      labelPicker = CP.create(holder, {
        entries: entries, selected: [], mode: 'compact', label: 'Class', emptyText: 'any',
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
    var alHolder = $('rv-al-picker');
    if (alHolder) {
      alPicker = CP.create(alHolder, {
        entries: entries, selected: catalog ? CU.defaultSelection(catalog) : visible(), mode: 'full', allowAdd: true,
        quickSets: catalog ? CU.quickSetsFor(catalog) : []
      });
    }
    var clsHolder = $('rv-classes-picker');
    if (clsHolder) {
      classes.forEach(function (c) { hideSet[c] = hiddenClasses.indexOf(c) >= 0; });
      classesPicker = CP.create(clsHolder, {
        entries: entries, selected: classes.slice(), mode: 'full', allowAdd: true,
        extra: function (entry, chip) {
          var lab = el('label', 'cp-extra');
          var cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.checked = !!hideSet[entry.name];
          cb.setAttribute('aria-label', 'Do not want ' + entry.name + ': hide it in Review and reject its pending boxes');
          cb.addEventListener('change', function () { hideSet[entry.name] = cb.checked; });
          lab.appendChild(cb);
          lab.appendChild(document.createTextNode("don't want (hide + reject)"));
          chip.appendChild(lab);
        }
      });
      var plain = $('rv-classes').elements.classes;
      if (plain) plain.hidden = true;
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
  loadNext();
})();
