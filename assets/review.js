// Review page (/review): one frame at a time from GET /v1/review/next, its boxes from
// GET /v1/review/labels, drawn on a canvas; decisions saved with POST /v1/review/labels.
// Filters (bucket, scene, camera, class, status, source) go to /v1/review/next as query params.
// Export / import / classes forms post JSON to /v1/review/{export,import,classes}.
// No inline handlers (CSP): everything is wired here; server values come from data-*.
(function () {
  'use strict';
  var root = document.getElementById('review');
  if (!root) return;
  var classes = [];
  try { classes = JSON.parse(root.getAttribute('data-classes') || '[]'); } catch (e) { classes = []; }

  var $ = function (id) { return document.getElementById(id); };
  var canvas = $('rv-canvas');
  var ctx = canvas.getContext('2d');
  var filtersForm = $('rv-filters');
  var classSel = $('rv-class');
  var boxList = $('rv-boxes');
  var emptyBox = $('rv-empty');
  var msg = $('rv-msg');

  var cur = null;       // answer of GET /v1/review/labels
  var img = null;       // the frame image
  var selected = -1;    // index into cur.labels.boxes
  var undo = [];        // snapshots of cur.labels
  var dirty = false;
  var drag = null;      // {x0, y0, x1, y1, moved} in frame pixels
  var busy = false;

  var COLORS = {
    pending: '#f0b400', hint: '#f08c00', accepted: '#2e9e4f', added: '#1f6fe0', rejected: '#e53935'
  };

  function say(text, isError) {
    msg.textContent = text || '';
    msg.className = 'form-msg' + (isError ? ' error-text' : '');
  }

  function api(url, opts) {
    return fetch(url, opts).then(function (r) {
      return r.json().catch(function () { return null; }).then(function (j) {
        if (!r.ok || !j || j.success === false) {
          throw new Error((j && j.message) || ('HTTP ' + r.status));
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

  function isHint(b) {
    return b.source === 'model' && b.origin && b.origin !== 'prediction';
  }

  // What a box shows as: the state, with pending split by what saving will make of it.
  function look(b) {
    if (b.state === 'pending') return isHint(b) ? 'hint' : 'pending';
    return b.state;
  }

  function fmtScore(s) { return (typeof s === 'number') ? s.toFixed(2) : ''; }

  // ---- drawing ------------------------------------------------------------------------------

  function lineWidth() { return Math.max(2, Math.round(canvas.width / 400)); }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img && img.complete && img.naturalWidth) {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
    if (!cur) return;
    var lw = lineWidth();
    var font = Math.max(12, Math.round(canvas.width / 70));
    ctx.font = font + 'px system-ui, sans-serif';
    ctx.textBaseline = 'bottom';
    boxes().forEach(function (b, i) {
      var l = look(b);
      var x = b.bbox[0], y = b.bbox[1], w = b.bbox[2] - b.bbox[0], h = b.bbox[3] - b.bbox[1];
      ctx.save();
      ctx.globalAlpha = l === 'rejected' ? 0.7 : 1;
      ctx.lineWidth = i === selected ? lw * 2 : lw;
      ctx.setLineDash(l === 'hint' || l === 'rejected' ? [lw * 4, lw * 3] : []);
      if (i === selected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = lw * 3.5;
        ctx.strokeRect(x, y, w, h);
        ctx.lineWidth = lw * 2;
      }
      ctx.strokeStyle = COLORS[l] || '#ffffff';
      ctx.strokeRect(x, y, w, h);
      if (l === 'rejected') {
        ctx.beginPath();
        ctx.moveTo(x, y); ctx.lineTo(x + w, y + h);
        ctx.moveTo(x + w, y); ctx.lineTo(x, y + h);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      var text = (i + 1) + ' ' + b.label + (b.score != null ? ' ' + fmtScore(b.score) : '');
      var tw = ctx.measureText(text).width + 8;
      var ty = y > font + 4 ? y : y + font + 4;
      ctx.fillStyle = COLORS[l] || '#ffffff';
      ctx.fillRect(x, ty - font - 4, tw, font + 4);
      ctx.fillStyle = '#000000';
      ctx.fillText(text, x + 4, ty - 2);
      ctx.restore();
    });
    if (drag && drag.moved) {
      ctx.save();
      ctx.lineWidth = lw;
      ctx.setLineDash([lw * 3, lw * 2]);
      ctx.strokeStyle = COLORS.added;
      var r = norm(drag);
      ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      ctx.restore();
    }
  }

  function norm(d) {
    return [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
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

  // The auto-labeler's evidence on a box: triage, why, the VLM answer, the top species.
  function autoNote(b) {
    var parts = [];
    if (b.triage) parts.push(b.triage === 'accept' ? 'auto: accept' : 'auto: review' + (b.why && b.why.length ? ' (' + b.why.join('; ') + ')' : ''));
    if (b.agree === false && !b.triage) parts.push('one labeler');
    if (b.vlm) parts.push((b.vlm.model || 'VLM') + ': ' + (b.vlm.answer || 'nothing') + (b.vlm.ok ? '' : ' (!)'));
    if (b.species_scores) {
      var best = null;
      Object.keys(b.species_scores).forEach(function (k) { if (!best || b.species_scores[k] > b.species_scores[best]) best = k; });
      if (best) parts.push('SigLIP: ' + best + ' ' + fmtScore(b.species_scores[best]));
    }
    return parts;
  }

  function note(b) {
    var parts = [b.source];
    if (b.origin) parts.push(b.origin.replace(/_/g, ' '));
    if (b.reason) parts.push(b.reason);
    if (b.score != null) parts.push('score ' + fmtScore(b.score));
    if (b.model_label && b.model_label !== b.label) parts.push('was ' + b.model_label);
    parts = parts.concat(autoNote(b));
    var st = b.state === 'pending'
      ? 'pending (' + (b.source === 'human' ? 'added' : (isHint(b) ? 'rejected' : 'accepted')) + ' on save)'
      : b.state;
    return st + ' · ' + parts.join(' · ');
  }

  function renderSide() {
    boxList.textContent = '';
    boxes().forEach(function (b, i) {
      var li = el('li', 'st-' + b.state + (i === selected ? ' sel' : ''));
      var sw = el('span', 'swatch sw-' + look(b));
      var sel = document.createElement('select');
      sel.className = 'bx-label';
      sel.setAttribute('aria-label', 'Class of box ' + (i + 1));
      var names = classes.slice();
      if (names.indexOf(b.label) < 0) names.unshift(b.label);
      names.forEach(function (c) {
        var o = el('option', null, c + (classes.indexOf(c) < 0 ? ' (not a class)' : ''));
        o.value = c;
        if (c === b.label) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener('click', function (ev) { ev.stopPropagation(); });
      sel.addEventListener('change', function () { relabel(i, sel.value); });
      var head = el('span', null, String(i + 1));
      head.appendChild(document.createTextNode(' '));
      head.insertBefore(sw, head.firstChild);
      li.appendChild(head);
      li.appendChild(sel);
      li.appendChild(el('span', 'bx-note', note(b)));
      var actions = el('span', 'bx-actions');
      if (b.source === 'human') {
        actions.appendChild(button('Delete', function () { removeBox(i); }));
      } else {
        if (b.state !== 'accepted') actions.appendChild(button('Accept', function () { setState(i, 'accepted'); }));
        if (b.state !== 'rejected') actions.appendChild(button('Reject', function () { setState(i, 'rejected'); }));
      }
      li.appendChild(actions);
      li.addEventListener('click', function () { select(i); });
      boxList.appendChild(li);
    });
    if (!boxes().length) boxList.appendChild(el('li', 'muted', cur ? 'No box. Drag on the image to draw one.' : ''));
    emptyBox.checked = !!(cur && cur.labels.empty_confirmed);
  }

  function refresh() { renderSide(); draw(); }

  // ---- edits --------------------------------------------------------------------------------

  function select(i) {
    selected = (i >= 0 && i < boxes().length) ? i : -1;
    refresh();
  }

  function setState(i, state) {
    var b = boxes()[i];
    if (!b) return;
    snapshot();
    b.state = state;
    if (state !== 'rejected' && cur.labels.empty_confirmed) cur.labels.empty_confirmed = false;
    refresh();
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

  function acceptAll() {
    if (!cur) return;
    snapshot();
    boxes().forEach(function (b) { if (b.state === 'pending') b.state = b.source === 'human' ? 'added' : 'accepted'; });
    cur.labels.empty_confirmed = false;
    refresh();
  }

  function confirmEmpty(on) {
    if (!cur) return;
    snapshot();
    cur.labels.empty_confirmed = on;
    if (on) {
      cur.labels.boxes = boxes().filter(function (b) { return b.source !== 'human'; });
      boxes().forEach(function (b) { b.state = 'rejected'; });
      selected = -1;
    }
    refresh();
  }

  function doUndo() {
    if (!cur || !undo.length) return;
    cur.labels = JSON.parse(undo.pop());
    if (selected >= boxes().length) selected = -1;
    refresh();
  }

  function pickClass(n) {
    var c = classes[n];
    if (c == null) return;
    if (selected >= 0) relabel(selected, c);
    else { classSel.value = c; say('New boxes: ' + c); }
  }

  // ---- canvas input -------------------------------------------------------------------------

  function framePoint(ev) {
    var r = canvas.getBoundingClientRect();
    var x = (ev.clientX - r.left) * canvas.width / r.width;
    var y = (ev.clientY - r.top) * canvas.height / r.height;
    return [Math.max(0, Math.min(canvas.width, x)), Math.max(0, Math.min(canvas.height, y))];
  }

  canvas.addEventListener('pointerdown', function (ev) {
    if (!cur || ev.button !== 0) return;
    var p = framePoint(ev);
    drag = { x0: p[0], y0: p[1], x1: p[0], y1: p[1], moved: false };
    canvas.setPointerCapture(ev.pointerId);
    canvas.focus();
  });
  canvas.addEventListener('pointermove', function (ev) {
    if (!drag) return;
    var p = framePoint(ev);
    drag.x1 = p[0]; drag.y1 = p[1];
    var min = Math.max(4, canvas.width / 200);
    if (Math.abs(drag.x1 - drag.x0) > min || Math.abs(drag.y1 - drag.y0) > min) drag.moved = true;
    if (drag.moved) draw();
  });
  canvas.addEventListener('pointerup', function () {
    if (!drag) return;
    var d = drag;
    drag = null;
    if (d.moved) {
      var r = norm(d);
      if (r[2] - r[0] >= 3 && r[3] - r[1] >= 3) {
        snapshot();
        boxes().push({
          label: classSel.value || classes[0] || 'object',
          bbox: r.map(function (v) { return Math.round(v * 10) / 10; }),
          source: 'human', state: 'added'
        });
        cur.labels.empty_confirmed = false;
        selected = boxes().length - 1;
      }
      refresh();
      return;
    }
    // A click: the smallest box under the point.
    var best = -1, area = Infinity;
    boxes().forEach(function (b, i) {
      var a = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
      if (d.x0 >= b.bbox[0] && d.x0 <= b.bbox[2] && d.y0 >= b.bbox[1] && d.y0 <= b.bbox[3] && a < area) {
        best = i; area = a;
      }
    });
    select(best);
  });
  canvas.addEventListener('pointercancel', function () { drag = null; draw(); });

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

  // The Class filter: the class list plus the classes the queue saw (model labels outside the
  // list too), the current choice kept.
  function fillLabels(list) {
    var sel = $('rv-label');
    if (!sel) return;
    var keep = sel.value;
    var have = Array.prototype.map.call(sel.options, function (o) { return o.value; });
    classes.concat(list || []).forEach(function (c) {
      if (have.indexOf(c) < 0) { var o = el('option', null, c); o.value = c; sel.appendChild(o); have.push(c); }
    });
    sel.value = keep;
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
    if (cur.stale_classes) parts.push('reviewed with another class list');
    if (cur.labels && cur.labels.disagreement) parts.push('back for review: ' + cur.labels.disagreement);
    m.textContent = parts.filter(Boolean).join(' · ');
  }

  // "Auto-label these frames": the filters of the queue become the job's frame filter.
  var alForm = $('rv-autolabel');
  if (alForm) {
    alForm.addEventListener('submit', function () {
      var parts = [];
      Array.prototype.forEach.call(filtersForm.elements, function (el) {
        if (el.name && el.value) parts.push(el.name + '=' + el.value);
      });
      var lim = alForm.elements.limit && alForm.elements.limit.value;
      if (lim) parts.push('limit=' + lim);
      alForm.elements.frames.value = parts.join(',');
    });
  }

  function showNone(on) {
    $('rv-none').hidden = !on;
    canvas.hidden = on;
  }

  function loadFrame(frame) {
    return api('/v1/review/labels?frame=' + encodeURIComponent(frame)).then(function (j) {
      cur = j;
      if (Array.isArray(j.classes)) classes = j.classes;
      undo = []; dirty = false; selected = -1;
      canvas.width = j.width || 640;
      canvas.height = j.height || 360;
      img = new Image();
      img.onload = draw;
      img.src = j.image_url;
      showNone(false);
      setMeta();
      refresh();
    });
  }

  function loadNext(after) {
    var p = filterParams();
    if (after) p.set('after', after);
    return api('/v1/review/next?' + p.toString()).then(function (j) {
      showCounts(j);
      fillCameras(j.cameras);
      fillLabels(j.labels);
      if (!j.item) {
        cur = null; img = null; undo = []; dirty = false;
        showNone(true); setMeta(); refresh();
        return;
      }
      return loadFrame(j.item.frame);
    }).catch(function (e) { say(e.message, true); });
  }

  function save(status) {
    if (!cur || busy) return;
    busy = true;
    var body = {
      version: 1,
      status: status,
      empty_confirmed: !!cur.labels.empty_confirmed,
      boxes: boxes(),
      classes_version: cur.labels.classes_version || ''
    };
    var frame = cur.frame;
    postJson('/v1/review/labels?frame=' + encodeURIComponent(frame), body).then(function (j) {
      say(j.message || 'Saved.');
      dirty = false;
      return loadNext(frame);
    }).catch(function (e) {
      say(e.message, true);
    }).then(function () { busy = false; });
  }

  function skipToNext() {
    if (!cur) return loadNext();
    if (dirty && !window.confirm('Leave this frame without saving your changes?')) return;
    say('');
    loadNext(cur.frame);
  }

  $('rv-save').addEventListener('click', function () { save('reviewed'); });
  $('rv-skip').addEventListener('click', function () { save('skipped'); });
  $('rv-next').addEventListener('click', skipToNext);
  $('rv-undo').addEventListener('click', doUndo);
  $('rv-accept-all').addEventListener('click', acceptAll);
  emptyBox.addEventListener('change', function () { confirmEmpty(emptyBox.checked); });
  filtersForm.addEventListener('change', function () { loadNext(); });
  filtersForm.addEventListener('submit', function (ev) { ev.preventDefault(); loadNext(); });

  document.addEventListener('keydown', function (ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    var t = ev.target;
    var tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
    if (tag === 'BUTTON' && ev.key === 'Enter') return;
    var k = ev.key;
    var handled = true;
    if (k === 'a') acceptAll();
    else if (k === 'r') {
      if (selected >= 0) setState(selected, boxes()[selected].state === 'rejected' ? 'accepted' : 'rejected');
    } else if (k === 'n') skipToNext();
    else if (k === 'e') confirmEmpty(!(cur && cur.labels.empty_confirmed));
    else if (k === 'z') doUndo();
    else if (k === 's' || k === 'Enter') save('reviewed');
    else if (k === 'j') select(boxes().length ? (selected + 1) % boxes().length : -1);
    else if (k === 'k') select(boxes().length ? (selected - 1 + boxes().length) % boxes().length : -1);
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

  // ---- export / import / classes ------------------------------------------------------------

  function list(text) {
    return String(text || '').split(/[,\n]/).map(function (s) { return s.trim(); }).filter(Boolean);
  }

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

  $('rv-classes').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var box = $('rv-classes-msg');
    postJson('/v1/review/classes', { classes: list(ev.target.elements.classes.value) }).then(function (j) {
      classes = j.classes;
      classSel.textContent = '';
      classes.forEach(function (c, i) {
        var o = el('option', null, (i + 1) + '. ' + c);
        o.value = c;
        classSel.appendChild(o);
      });
      box.textContent = j.message || 'Saved.';
      box.className = 'form-msg ok-text';
      refresh();
    }).catch(function (e) { box.textContent = e.message; box.className = 'form-msg error-text'; });
  });

  loadNext();
})();
