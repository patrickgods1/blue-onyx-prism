// Review page (/review): one frame at a time from GET /v1/review/next, its boxes from
// GET /v1/review/labels, drawn on a canvas; decisions saved with POST /v1/review/labels
// (with the loaded `revision`; a 409 means someone else saved the frame meanwhile).
// Filters (bucket, scene, camera, class, auto, status, source) go to /v1/review/next as query
// params. The canvas edits boxes (draw, select, move, resize, nudge) under a zoom / pan view.
// The Auto-label card queues a job (POST /training/autolabel as JSON) and follows it on
// GET /v1/training; "Accept auto: accept frames" calls POST /v1/review/accept-auto and
// "Reject auto boxes below" POST /v1/review/reject-low (both: dry run, confirm, run).
// Export / import / classes / folder forms post JSON to /v1/review/{export,import,classes,folder};
// a new folder reloads the page.
// Classes: GET /v1/review/class-catalog feeds the class pickers (class-picker.js): the Class
// filter (label=a,b), "Show boxes of" (display only), the auto-label form (repeated `classes`
// fields; a class outside the list is appended after a confirm) and the Classes card (list in
// order, "hide in Review" per class). Pickers, the new-box class select and the 1-0 keys use the
// class list minus the hidden classes; saved labels and class ids never change with that.
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

  var cur = null;       // answer of GET /v1/review/labels (with `revision`)
  var img = null;       // the frame image
  var selected = -1;    // index into cur.labels.boxes
  var undo = [];        // snapshots of cur.labels
  var dirty = false;
  var gesture = null;   // the pointer gesture in progress (see the canvas input section)
  var busy = false;
  var view = { z: 1, ox: 0, oy: 0 };   // zoom and the frame point at the canvas's top left
  var hideBoxes = false;               // `h` held
  var showTags = true;                 // `l` toggles
  var spaceDown = false;               // Space held: left-drag pans
  var overCanvas = false;
  var guard = null;                    // {frame, count}: the save that warned about review boxes
  var emptyBackup = null;              // {frame, boxes}: the boxes before "Confirmed empty"
  var upcoming = [];                   // the next frames of the queue (prefetch)
  var prefetchImg = null;
  var seenLabels = [];                 // classes the queue saw (Class filter)
  var lastNudge = null;                // {i, t}: arrow nudges in a row share one undo step
  var labelPicker = null, showPicker = null, alPicker = null, classesPicker = null;

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
    undo.push(JSON.stringify(cur.labels));
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

  // A pending box the auto-labeler flagged: triage `review`, or it names another class.
  function isReview(b) {
    return b.state === 'pending' && (b.triage === 'review' || (!!b.auto_label && b.auto_label !== b.label));
  }

  // What a box shows as: the state, with pending split by review / what saving will make of it.
  function look(b) {
    if (b.state === 'pending') return isReview(b) ? 'review' : (isHint(b) ? 'hint' : 'pending');
    return b.state;
  }

  function reviewCount() { return boxes().filter(isReview).length; }

  function firstReview() {
    var bs = boxes();
    for (var i = 0; i < bs.length; i++) if (isReview(bs[i])) return i;
    return -1;
  }

  // The box to look at after `i` (wrapping; -1 for the first): the next review box, else the
  // next pending one, else -1.
  function nextUndecided(i) {
    var bs = boxes();
    var tests = [isReview, function (b) { return b.state === 'pending'; }];
    for (var t = 0; t < tests.length; t++) {
      for (var k = 1; k <= bs.length; k++) {
        var j = (i + k + bs.length) % bs.length;
        if (j !== i && tests[t](bs[j])) return j;
      }
    }
    return -1;
  }

  function fmtScore(s) { return (typeof s === 'number') ? s.toFixed(2) : ''; }

  // The classes Review offers: the list minus the hidden ones.
  function visible() { return classes.filter(function (c) { return hiddenClasses.indexOf(c) < 0; }); }

  // "Show boxes of": a box of another class is dimmed (display only; saving keeps every box).
  function isShown(b) { return !shownLabels.length || shownLabels.indexOf(b.label) >= 0; }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function round1(v) { return Math.round(v * 10) / 10; }

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
    view.z = 1; view.ox = 0; view.oy = 0;
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
    if (zoomText) zoomText.textContent = zoomed ? Math.round(view.z * 100) + '%' : '';
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
    return (l === 'review' ? '? ' : '') + (i + 1) + ' ' + b.label +
      (b.score != null ? ' ' + fmtScore(b.score) : '') +
      (b.source === 'auto' && b.triage === 'accept' && b.state !== 'rejected' ? ' ✓' : '');
  }

  function draw() {
    var W = canvas.width, H = canvas.height;
    clampView();
    updateViewUi();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    var z = view.z;
    ctx.setTransform(z, 0, 0, z, -view.ox * z, -view.oy * z);
    if (img && img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, W, H);
    drawCrop();
    if (!cur || hideBoxes) return;
    var lw = lineWidth() / z;
    var font = Math.max(12, Math.round(W / 70)) / z;
    var pad = 4 / z;
    var th = font + pad;
    ctx.font = font + 'px system-ui, sans-serif';
    ctx.textBaseline = 'bottom';
    boxes().forEach(function (b, i) {
      var l = look(b);
      var x = b.bbox[0], y = b.bbox[1], w = b.bbox[2] - b.bbox[0], h = b.bbox[3] - b.bbox[1];
      var color = COLORS[l] || '#ffffff';
      var dashed = l === 'rejected' || isHint(b);
      ctx.save();
      var shown = isShown(b);
      ctx.globalAlpha = (l === 'rejected' ? 0.7 : 1) * (shown ? 1 : 0.2);
      if (i === selected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = lw * 3.5;
        ctx.strokeRect(x, y, w, h);
      }
      ctx.setLineDash(dashed ? [lw * 4, lw * 3] : []);
      ctx.lineWidth = (i === selected ? 2 : 1) * (l === 'review' ? 1.5 : 1) * lw;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
      if (l === 'review') {
        // A dark inner line: the flagged boxes read as outlined.
        ctx.setLineDash([]);
        ctx.lineWidth = lw * 0.6;
        ctx.strokeStyle = '#000000';
        var inset = (i === selected ? 2.25 : 1.1) * lw;
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
      if (showTags && shown) {
        var text = tagText(b, i);
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
    });
    var sb = boxes()[selected];
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

  // The selected box with a 25% margin, upscaled, in the side panel (to check the evidence).
  function drawCrop() {
    if (!cropCanvas) return;
    var b = boxes()[selected];
    if (!b || !img || !img.complete || !img.naturalWidth) { cropWrap.hidden = true; return; }
    var W = canvas.width, H = canvas.height;
    var bw = b.bbox[2] - b.bbox[0], bh = b.bbox[3] - b.bbox[1];
    var x0 = clamp(b.bbox[0] - bw * 0.25, 0, W), y0 = clamp(b.bbox[1] - bh * 0.25, 0, H);
    var x1 = clamp(b.bbox[2] + bw * 0.25, 0, W), y1 = clamp(b.bbox[3] + bh * 0.25, 0, H);
    var sw = Math.max(1, x1 - x0), sh = Math.max(1, y1 - y0);
    var scale = Math.min(288 / sw, 200 / sh, 8);
    var cw = Math.max(1, Math.round(sw * scale)), ch = Math.max(1, Math.round(sh * scale));
    if (cropCanvas.width !== cw) cropCanvas.width = cw;
    if (cropCanvas.height !== ch) cropCanvas.height = ch;
    var c = cropCanvas.getContext('2d');
    var kx = img.naturalWidth / W, ky = img.naturalHeight / H;
    c.clearRect(0, 0, cw, ch);
    c.drawImage(img, x0 * kx, y0 * ky, sw * kx, sh * ky, 0, 0, cw, ch);
    c.lineWidth = 2;
    c.strokeStyle = COLORS[look(b)] || '#ffffff';
    c.strokeRect((b.bbox[0] - x0) * scale, (b.bbox[1] - y0) * scale, bw * scale, bh * scale);
    cropCap.textContent = 'Box ' + (selected + 1) + ' (' + b.label + '), ' + Math.round(bw) + ' x ' + Math.round(bh) + ' px';
    cropWrap.hidden = false;
  }

  // ---- side panel ---------------------------------------------------------------------------

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function button(text, onClick) {
    var b = el('button', 'small', text);
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
    sel.addEventListener('change', function () { relabel(i, sel.value); });
    return sel;
  }

  function boxItem(b, i) {
    var li = el('li', 'st-' + b.state + ' lk-' + look(b) + (i === selected ? ' sel' : ''));
    var head = el('span', null);
    head.appendChild(el('span', 'swatch sw-' + look(b)));
    head.appendChild(document.createTextNode((isReview(b) ? '? ' : '') + String(i + 1) + ' '));
    li.appendChild(head);
    li.appendChild(classSelect(b, i));
    var chips = el('span', 'bx-chips');
    chips.appendChild(stateChip(b));
    var t = triageChip(b);
    if (t) chips.appendChild(t);
    li.appendChild(chips);
    var ev = evidence(b);
    if (ev) li.appendChild(el('span', 'bx-note', ev));
    if (b.auto_label && b.auto_label !== b.label) {
      var sug = el('span', 'bx-suggest');
      sug.appendChild(document.createTextNode('Auto-labelers say ' + b.auto_label +
        (b.auto_score != null ? ' (' + fmtScore(b.auto_score) + ')' : '') + ' '));
      var al = b.auto_label;
      sug.appendChild(button('Use ' + al, function () { relabel(i, al); }));
      li.appendChild(sug);
    }
    var actions = el('span', 'bx-actions');
    if (b.source === 'human') {
      actions.appendChild(button('Delete', function () { removeBox(i); }));
    } else {
      if (b.state !== 'accepted') actions.appendChild(button('Accept', function () { setState(i, 'accepted'); }));
      if (b.state !== 'rejected') actions.appendChild(button('Reject', function () { setState(i, 'rejected'); }));
    }
    li.appendChild(actions);
    li.addEventListener('click', function () { select(i); });
    return li;
  }

  function renderSide() {
    boxList.textContent = '';
    var bs = boxes();
    // Pending review boxes first; each keeps its number (the one on the canvas).
    var order = [];
    bs.forEach(function (b, i) { if (isReview(b)) order.push(i); });
    bs.forEach(function (b, i) { if (!isReview(b)) order.push(i); });
    var offScreen = 0;
    order.forEach(function (i) {
      if (!isShown(bs[i])) { offScreen++; return; }
      boxList.appendChild(boxItem(bs[i], i));
    });
    if (offScreen) {
      boxList.appendChild(el('li', 'muted', offScreen + ' box(es) of other classes hidden by "Show boxes of" (still saved).'));
    }
    if (!bs.length && cur) boxList.appendChild(el('li', 'muted', 'No box. Drag on the image to draw one.'));
    emptyBox.checked = !!(cur && cur.labels.empty_confirmed);
  }

  function refresh() { renderSide(); draw(); }

  // Scrolls the box list so the selected box is in view; `page`: the page too, as little as needed
  // (not on a canvas click, which would move the frame under the pointer).
  function revealSelected(page) {
    var li = boxList.querySelector('li.sel');
    if (!li) return;
    if (page) {
      // The enlarged crop sits above the list: show it, then the entry.
      if (cropWrap && !cropWrap.hidden) cropWrap.scrollIntoView({ block: 'nearest' });
      li.scrollIntoView({ block: 'nearest' });
      return;
    }
    var lr = li.getBoundingClientRect(), br = boxList.getBoundingClientRect();
    if (lr.top < br.top) boxList.scrollTop += lr.top - br.top - 4;
    else if (lr.bottom > br.bottom) boxList.scrollTop += lr.bottom - br.bottom + 4;
  }

  // A new frame: the page shows the frame from its top, the box list starts at its top.
  function revealFrame() {
    boxList.scrollTop = 0;
    revealSelected();
    var grid = canvas.closest('.review-grid');
    if (grid && grid.getBoundingClientRect().top < 0) {
      grid.scrollIntoView({ block: 'start' });
      revealSelected(true);
    }
  }

  // ---- edits --------------------------------------------------------------------------------

  function select(i, page) {
    selected = (i >= 0 && i < boxes().length) ? i : -1;
    panTo(boxes()[selected]);
    refresh();
    revealSelected(page);
  }

  // Deciding a box moves the selection on to the next box to review (if any).
  function setState(i, state) {
    var b = boxes()[i];
    if (!b) return;
    snapshot();
    b.state = state;
    if (state !== 'rejected' && cur.labels.empty_confirmed) cur.labels.empty_confirmed = false;
    var next = nextUndecided(i);
    select(next >= 0 ? next : i, true);
  }

  function relabel(i, label) {
    var b = boxes()[i];
    if (!b || b.label === label) return;
    snapshot();
    if (b.source !== 'human' && !b.model_label) b.model_label = b.label;
    b.label = label;
    if (b.source !== 'human') b.state = 'accepted';
    if (cur.labels.empty_confirmed) cur.labels.empty_confirmed = false;
    refresh();
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
      return b.state === 'pending' && b.source !== 'human' && resolved(b) === 'accepted';
    });
    var hints = boxes().filter(isHint).length;
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
      boxes().forEach(function (b) { b.state = 'rejected'; });
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
    cur.labels = JSON.parse(undo.pop());
    lastNudge = null;
    if (selected >= boxes().length) selected = -1;
    refresh();
  }

  function pickClass(n) {
    var c = visible()[n];
    if (c == null) return;
    if (selected >= 0) relabel(selected, c);
    else { classSel.value = c; say('New boxes: ' + c); }
  }

  // The first geometric edit of a model / auto box keeps its original box and accepts it.
  function markEdited(b) {
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

  function hoverCursor(ev) {
    if (gesture) return;
    if (spaceDown) { setCursor('grab'); return; }
    if (!cur || hideBoxes || !ev) { setCursor('crosshair'); return; }
    var p = framePoint(ev);
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
            source: 'human', state: 'added'
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
    var best = -1, area = Infinity;
    if (!hideBoxes) {
      boxes().forEach(function (bx, i) {
        var a = (bx.bbox[2] - bx.bbox[0]) * (bx.bbox[3] - bx.bbox[1]);
        if (isShown(bx) && g.x0 >= bx.bbox[0] && g.x0 <= bx.bbox[2] && g.y0 >= bx.bbox[1] && g.y0 <= bx.bbox[3] && a < area) {
          best = i; area = a;
        }
      });
    }
    select(best);
    hoverCursor(ev);
  });

  canvas.addEventListener('pointercancel', function () {
    var g = gesture;
    gesture = null;
    // A cancelled move / resize keeps what it did (it is one undo step).
    if (g && g.moved && g.mode !== 'draw' && g.mode !== 'pan') refresh(); else draw();
  });
  canvas.addEventListener('pointerenter', function () { overCanvas = true; });
  canvas.addEventListener('pointerleave', function () { overCanvas = false; });
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
    var c = canvasPoint(ev);
    zoomAt(c[0], c[1], factor);
  }, { passive: false });
  if (fitBtn) fitBtn.addEventListener('click', function () { fit(); canvas.focus(); });

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
    var k = $('rv-key-classes');
    if (!k) return;
    var shown = visible().slice(0, 10).map(function (c, i) { return ((i + 1) % 10) + ' ' + c; });
    k.textContent = shown.length ? ' (' + shown.join(', ') + ')' : '';
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

  function loadFrame(frame) {
    return api('/v1/review/labels?frame=' + encodeURIComponent(frame)).then(function (j) {
      cur = j;
      if (Array.isArray(j.classes)) classes = j.classes;
      undo = []; dirty = false; guard = null; lastNudge = null; gesture = null;
      emptyBackup = null;
      canvas.width = j.width || 640;
      canvas.height = j.height || 360;
      view.z = 1; view.ox = 0; view.oy = 0;
      selected = nextUndecided(-1);
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
    }
    busy = true;
    var body = JSON.parse(JSON.stringify(cur.labels));   // keeps auto_run, bulk_accepted, ...
    body.version = body.version || 1;
    body.status = status;
    body.empty_confirmed = !!cur.labels.empty_confirmed;
    body.classes_version = cur.labels.classes_version || '';
    var frame = cur.frame;
    var url = '/v1/review/labels?frame=' + encodeURIComponent(frame);
    if (cur.revision != null) url += '&revision=' + encodeURIComponent(cur.revision);
    postJson(url, body).then(function (j) {
      dirty = false;
      return loadNext(frame, j.message || 'Saved.');
    }).catch(function (e) {
      if (e.conflict) showConflict(e.message);
      else say(e.message, true);
    }).then(function () { busy = false; });
  }

  function skipToNext() {
    if (!cur) return loadNext();
    if (dirty && !window.confirm('Leave this frame without saving your changes?')) return;
    loadNext(cur.frame, '');
  }

  $('rv-save').addEventListener('click', function () { save('reviewed'); });
  $('rv-skip').addEventListener('click', function () { save('skipped'); });
  $('rv-next').addEventListener('click', skipToNext);
  $('rv-undo').addEventListener('click', doUndo);
  $('rv-accept-all').addEventListener('click', acceptProposals);
  emptyBox.addEventListener('change', function () { confirmEmpty(emptyBox.checked); });
  var filterTimer = null;
  function filterChanged() {
    if (filterTimer) clearTimeout(filterTimer);
    filterTimer = setTimeout(function () { filterTimer = null; loadNext(); }, 200);
  }
  filtersForm.addEventListener('change', function (ev) {
    // The class picker reports through its own onChange (one reload per change).
    if (ev.target && ev.target.closest && ev.target.closest('.cp')) return;
    loadNext();
  });
  filtersForm.addEventListener('submit', function (ev) { ev.preventDefault(); loadNext(); });

  function typing(t) {
    var tag = t && t.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || !!(t && t.isContentEditable);
  }

  document.addEventListener('keydown', function (ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    var t = ev.target;
    if (typing(t)) return;
    // The Browse dialog is modal: its keys are not review shortcuts.
    if (document.querySelector('dialog[open]')) return;
    var k = ev.key;
    if (t && (t.tagName === 'BUTTON' || t.tagName === 'SUMMARY') && (k === 'Enter' || k === ' ')) return;
    var handled = true;
    if (k === ' ') {
      spaceDown = true;
      if (!gesture) setCursor('grab');
      // Space scrolls the page unless the pointer or the focus is on the frame.
      handled = overCanvas || document.activeElement === canvas;
    } else if (k === 'h') {
      if (!ev.repeat && !hideBoxes) { hideBoxes = true; draw(); }
    } else if (k === 'l') { showTags = !showTags; draw(); }
    else if (k === 'f') fit();
    else if (k === 'ArrowLeft' || k === 'ArrowRight' || k === 'ArrowUp' || k === 'ArrowDown') {
      var step = ev.shiftKey ? 10 : 1;
      handled = selected >= 0 && nudge(
        k === 'ArrowLeft' ? -step : (k === 'ArrowRight' ? step : 0),
        k === 'ArrowUp' ? -step : (k === 'ArrowDown' ? step : 0));
    } else if (k === 'a') acceptProposals();
    else if (k === 'r') {
      if (selected >= 0) setState(selected, boxes()[selected].state === 'rejected' ? 'accepted' : 'rejected');
    } else if (k === 'n') skipToNext();
    else if (k === 'e') confirmEmpty(!(cur && cur.labels.empty_confirmed));
    else if (k === 'z') doUndo();
    else if (k === 's' || k === 'Enter') save('reviewed');
    else if (k === 'j') select(boxes().length ? (selected + 1) % boxes().length : -1, true);
    else if (k === 'k') select(boxes().length ? (selected - 1 + boxes().length) % boxes().length : -1, true);
    else if (k === 'Delete' || k === 'Backspace') {
      if (selected >= 0) {
        if (boxes()[selected].source === 'human') removeBox(selected);
        else setState(selected, 'rejected');
      }
    } else if (k === 'Escape') select(-1);
    else if (/^[0-9]$/.test(k)) pickClass(k === '0' ? 9 : Number(k) - 1);
    else handled = false;
    if (handled) ev.preventDefault();
  });

  document.addEventListener('keyup', function (ev) {
    if (ev.key === 'h' && hideBoxes) { hideBoxes = false; draw(); }
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
      var ready = alPicker && catalog ? CP.ensureInList(picked, catalog) : Promise.resolve(catalog);
      report(alStatus, ['Queueing...']);
      ready.then(function () {
        if (alPicker) { refreshPickers(); }
        return api('/training/autolabel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json' },
          body: body.toString()
        });
      }).then(function (j) {
        var m = j.message || 'Queued.';
        report(alStatus, [/^Queued/.test(m) ? m : 'Queued: ' + m]);
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
    var box = $('rv-folder-msg');
    box.textContent = 'Switching...';
    box.className = 'form-msg';
    postJson('/v1/review/folder', { dir: dir }).then(function () {
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
        'YOLO class ids of later exports change (frames you exported before keep the old ids), and reviewed frames ' +
        'labelled with a removed class keep their labels but are marked as reviewed with another class list. Continue?')) {
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
      box.textContent = j.message || 'Saved.';
      box.className = 'form-msg ok-text';
    }).catch(function (e) { box.textContent = e.message; box.className = 'form-msg error-text'; });
  });

  // ---- class pickers ------------------------------------------------------------------------

  var hideSet = {};   // Classes card: class -> "hide in Review" ticked

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
          cb.setAttribute('aria-label', 'Hide ' + entry.name + ' in Review');
          cb.addEventListener('change', function () { hideSet[entry.name] = cb.checked; });
          lab.appendChild(cb);
          lab.appendChild(document.createTextNode('hide in Review'));
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
