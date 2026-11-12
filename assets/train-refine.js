// Refine tab (/train/refine): starts the dataset check and shows what it found. The Check card
// previews (GET /v1/refine/preview) and starts (POST /v1/refine/check) a check and follows it
// through PrismPoll on GET /v1/train/jobs?tasks=busy (every 3 s while one is listed, every 30 s
// while none is, so a check started elsewhere shows up), with Pause / Resume / Cancel posted to the
// queue's routes. The Findings and Coverage cards are drawn from GET /v1/refine (the report, the
// open suggestions per kind) and read again when a check ends. The check changes no box, so
// nothing here confirms a box: Review does. The one question asked here is whether a start may
// replace the open suggestions of the frames it checks (inline, after the server's 409). The
// Static objects card (GET /v1/refine/spots) lists the places where one parked object sits across
// a camera's frames, with the one bulk action of this tab: "Accept the N at this spot" adds the
// check's missing boxes there (POST /v1/refine/accept-spot: a dry run first, then the inline
// question, then the write; the answer names frames to look at in Review) and the training cap
// (POST /v1/refine/static-cap). No inline handlers (CSP); limits come from the inputs' min/max and
// the server's answers.
(function () {
  'use strict';
  var root = document.getElementById('refine-tab');
  if (!root) return;
  var $ = function (id) { return document.getElementById(id); };
  var form = $('refine-form');
  var KINDS = [
    { kind: 'add', title: 'Missing boxes', text: 'Objects a fold model found that no box labels.', key: 'add' },
    { kind: 'verify', title: 'Labels no model found', text: 'Boxes the fold models never found: look at them, some are wrong.', key: 'verify' },
    { kind: 'relabel', title: 'Other class suggested', text: 'Boxes a fold model would give another class.', key: 'relabel' }
  ];

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function say(text, bad) {
    var box = $('refine-msg');
    box.textContent = text || '';
    box.className = bad ? 'form-msg error' : 'form-msg';
  }

  function getJson(url) {
    return fetch(url, { headers: { Accept: 'application/json' }, cache: 'no-store' }).then(parse);
  }
  function postJson(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body)
    }).then(parse);
  }
  function parse(r) {
    return r.json().catch(function () { return null; }).then(function (j) {
      if (!r.ok || !j || j.success === false) {
        var e = new Error((j && j.message) || ('HTTP ' + r.status));
        e.body = j;
        throw e;
      }
      return j;
    });
  }

  function pct(x) { return Math.round(x * 100) + '%'; }
  // A score or IoU as the page prints it: two decimals, trailing zeros trimmed (0.3, 0.5, 0.15).
  // The report's f32 thresholds arrive widened to f64 (0.30000001192092896); never print them raw.
  function num(x) {
    var v = x == null ? NaN : Number(x);
    return isFinite(v) ? String(Number(v.toFixed(2))) : '?';
  }
  function seconds(s) {
    if (s < 90) return 'under 2 min';
    var m = Math.round(s / 60);
    if (m < 90) return 'about ' + m + ' min';
    return 'about ' + (Math.round(s / 360) / 10) + ' h';
  }
  function when(iso) {
    var d = new Date(iso);
    return isNaN(d.getTime()) ? String(iso || '') : d.toLocaleString();
  }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }
  function list(text) {
    return String(text || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean);
  }
  function whole(v) {
    var s = String(v == null ? '' : v).trim();
    return s === '' ? null : Number(s);
  }

  // ---- the form

  function values() {
    var f = form.elements;
    var o = {
      folds: whole(f.folds.value),
      epochs: whole(f.epochs.value),
      scenes: f.scene.value ? [f.scene.value] : [],
      cameras: list(f.camera.value),
      allow_large: !!f.allow_large.checked
    };
    if (f.gpu && f.gpu.value) o.gpu = f.gpu.value;
    return o;
  }
  function query(o) {
    var p = [];
    var add = function (k, v) { p.push(encodeURIComponent(k) + '=' + encodeURIComponent(v)); };
    if (o.folds != null) add('folds', o.folds);
    if (o.epochs != null) add('epochs', o.epochs);
    if (o.scenes.length) add('scene', o.scenes.join(','));
    if (o.cameras.length) add('camera', o.cameras.join(','));
    if (o.gpu) add('gpu', o.gpu);
    if (o.allow_large) add('allow_large', '1');
    return p.join('&');
  }

  function drawPreview(p) {
    var box = $('refine-preview');
    clear(box);
    box.hidden = false;
    box.appendChild(el('p', null, 'Would check ' + plural(p.frames, 'frame', 'frames') + ' on ' +
      plural(p.camera_days, 'camera day', 'camera days') + ' (' + p.filter + ').'));
    var l = p.left_out || {};
    var out = [];
    if (l.pending) out.push(l.pending + ' with boxes still pending');
    if (l.off_class_list) out.push(l.off_class_list + ' with labels off the class list');
    if (l.lost) out.push(l.lost + ' with a file gone');
    if (l.unreadable) out.push(l.unreadable + ' unreadable');
    if (l.filtered) out.push(l.filtered + ' outside the scene or camera');
    if (out.length) box.appendChild(el('p', 'muted small', 'Left out: ' + out.join(', ') + '.'));
    if (p.folds && p.folds.length) {
      var r = p.recipe || {};
      box.appendChild(el('p', null, plural(p.folds.length, 'fold', 'folds') + ', each a ' + r.model +
        ' model at ' + r.imgsz + ' px trained ' + plural(r.epochs, 'epoch', 'epochs') + ' (batch ' + r.batch +
        '; from ' + r.from + ').'));
      var ul = el('ul', 'small');
      p.folds.forEach(function (f) {
        ul.appendChild(el('li', null, 'Fold ' + f.fold + ': ' + plural(f.frames, 'frame', 'frames') + ' on ' +
          plural(f.camera_days, 'camera day', 'camera days') + ' (the other folds train its model)'));
      });
      box.appendChild(ul);
    }
    if (p.seconds != null) {
      // time_basis: last_run = the pace the training run the recipe came from measured on this
      // GPU; else the generic estimate.
      var basis = p.time_basis === 'last_run' ? 'from your last training run' : 'a rough guess';
      box.appendChild(el('p', null, 'Time: ' + seconds(p.seconds) + ' on this GPU (' + basis + '). The GPU is busy meanwhile.'));
    }
    if (p.warnings && p.warnings.length) {
      var w = el('ul', 'warn-list small');
      p.warnings.forEach(function (t) { w.appendChild(el('li', null, t)); });
      box.appendChild(w);
    }
    if (p.refused) box.appendChild(el('p', 'error', 'A start would be refused: ' + p.refused));
    else box.appendChild(el('p', 'muted small', 'Nothing was run or written.'));
  }

  $('refine-preview-btn').addEventListener('click', function () {
    var btn = $('refine-preview-btn');
    btn.disabled = true;
    say('');
    getJson('/v1/refine/preview?' + query(values()))
      .then(drawPreview)
      .catch(function (e) { $('refine-preview').hidden = true; say(e.message, true); })
      .then(function () { btn.disabled = false; });
  });

  // A check replaces the open suggestions of the frames it checks. The server refuses a start
  // over open ones (409, needs_confirm) until it is asked again with replace_open: the second
  // step is inline, below the form (no dialog), and goes away when a field changes.
  function hideConfirm() { $('refine-confirm').hidden = true; }
  function askReplace(b) {
    $('refine-confirm-text').textContent = b.message;
    $('refine-confirm-start').textContent = 'Start and replace ' +
      plural(b.open_suggestions, 'open suggestion', 'open suggestions');
    say('');
    $('refine-confirm').hidden = false;
    $('refine-confirm-start').focus();
  }

  function start(replace) {
    var btn = $('refine-start');
    var again = $('refine-confirm-start');
    btn.disabled = true;
    again.disabled = true;
    hideConfirm();
    say('Starting...');
    var body = values();
    if (replace) body.replace_open = true;
    postJson('/v1/refine/check', body)
      .then(function (j) {
        say(j.message);
        $('refine-preview').hidden = true;
        window.dispatchEvent(new Event('train-area:refresh'));
        poller.kick();
        load();
      })
      .catch(function (e) {
        if (!replace && e.body && e.body.needs_confirm) askReplace(e.body);
        else say(e.message, true);
      })
      .then(function () { btn.disabled = !canCheck; again.disabled = false; });
  }

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    start(false);
  });
  $('refine-confirm-start').addEventListener('click', function () { start(true); });
  $('refine-confirm-keep').addEventListener('click', function () {
    hideConfirm();
    say('Not started: the open suggestions stay. Answer them on the Review tab, then start the check.');
  });
  form.addEventListener('input', hideConfirm);
  form.addEventListener('change', hideConfirm);

  // ---- the check in the queue

  var canCheck = true;
  var jobId = null;
  var parts = null;

  // Pause / Resume / Cancel. A refusal or a network error is said; the job row follows the queue
  // either way (the next poll shows what the queue did).
  function post(path, id, what) {
    return fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'text/html' },
      body: 'id=' + encodeURIComponent(id) + '&tab=refine'
    }).then(function (r) {
      if (!r.ok) say(what + ' did not go through (HTTP ' + r.status + '). Try again, or see the Jobs tab.', true);
    }, function (e) {
      say(what + ' was not sent: ' + ((e && e.message) || 'the network failed') + '. Try again.', true);
    }).then(function () { poller.kick(); });
  }
  function button(label, cls, run) {
    var b = el('button', cls, label);
    b.type = 'button';
    var enable = function () { b.disabled = false; };
    b.addEventListener('click', function () {
      b.disabled = true;
      var p;
      try { p = Promise.resolve(run()); } catch (e) { p = Promise.reject(e); }
      p.then(enable, function (e) { enable(); say((e && e.message) || String(e), true); });
    });
    return b;
  }

  function buildJob(j) {
    var box = $('jobs');
    clear(box);
    var row = el('div', 'refine-job');
    row.id = 'job-' + j.id;
    var head = el('p', null, '');
    var title = el('strong', null, j.title);
    var state = el('span', 'muted small', '');
    head.appendChild(title);
    head.appendChild(document.createTextNode(' '));
    head.appendChild(state);
    var progress = el('progress');
    progress.max = 100;
    var detail = el('p', 'muted small', '');
    var time = el('p', 'muted small', '');
    var bar = el('div', 'inline-row');
    var pause = button('Pause', 'small', function () { return post('/training/pause', j.id, 'Pause'); });
    var resume = button('Resume', 'small', function () { return post('/training/resume', j.id, 'Resume'); });
    var cancel = button('Cancel', 'small danger', function () {
      if (!window.confirm('Cancel the check? Folds that finished keep their suggestions.')) return Promise.resolve();
      return post('/training/cancel', j.id, 'Cancel');
    });
    [pause, resume, cancel].forEach(function (b) { bar.appendChild(b); });
    [head, progress, detail, time, bar].forEach(function (n) { row.appendChild(n); });
    box.appendChild(row);
    box.hidden = false;
    parts = { state: state, progress: progress, detail: detail, time: time, pause: pause, resume: resume };
    jobId = j.id;
    if (location.hash === '#job-' + j.id && row.scrollIntoView) row.scrollIntoView();
  }

  function drawJob(j) {
    if (jobId !== j.id || !parts) buildJob(j);
    var step = [j.step, j.phase].filter(Boolean).join(': ');
    parts.state.textContent = j.queued ? 'waiting for its turn' : (j.paused ? 'paused' : 'running');
    parts.progress.value = j.percent;
    parts.detail.textContent = [step, j.detail].filter(Boolean).join(' - ');
    parts.time.textContent = j.time || '';
    parts.pause.hidden = j.paused || j.queued;
    parts.resume.hidden = !j.paused;
  }

  function clearJob() {
    var box = $('jobs');
    clear(box);
    box.hidden = true;
    jobId = null;
    parts = null;
  }

  // The jobs list is the cheap poll (memory only). A check gone from it has ended: the findings
  // and coverage are read again, once.
  function follow() {
    return fetch('/v1/train/jobs?tasks=busy', { headers: { Accept: 'application/json' }, cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.jobs) return true;
        var mine = j.jobs.active.concat(j.jobs.queued).filter(function (x) {
          return x.link && x.link.indexOf('/train/refine#job-') === 0;
        })[0];
        if (mine) { drawJob(mine); return true; }
        var had = jobId !== null;
        clearJob();
        if (had) {
          load();
          window.dispatchEvent(new Event('train-area:refresh'));
        }
        return false;
      });
  }
  var poller = window.PrismPoll.create({ run: follow, period: 3000, more: function (again) { return again; } });
  // The 3 s poll parks itself when no check is listed. While it is parked, one cheap read every
  // 30 s (none while the tab is hidden), so a check started from another tab or the CLI shows
  // without a reload; finding one hands it back to the 3 s poll. Its first run is the page's own
  // load (GET /v1/refine carries the check), so it starts with a wait.
  var idleRuns = 0;
  window.PrismPoll.create({
    period: 30000,
    run: function () {
      if (idleRuns++ === 0 || !poller.state().parked) return null;
      return follow().then(function () { if (jobId !== null) poller.kick(); });
    }
  }).start();

  // ---- findings and coverage

  function reviewLink(kind) {
    var a = el('a', 'small', 'Review');
    a.href = '/train/review?auto=refine&refine_kind=' + encodeURIComponent(kind);
    return a;
  }

  function drawFindings(state) {
    var box = $('refine-findings-body');
    clear(box);
    var r = state.report, open = state.open;
    if (state.folder_missing) {
      box.appendChild(el('p', 'error', state.message || 'The folder under review is not available.'));
      return;
    }
    if (state.report_problem) box.appendChild(el('p', 'error', state.report_problem));
    if (state.last && state.last.state === 'failed' && state.last.error && !state.job) {
      box.appendChild(el('p', 'error', 'The last check failed: ' + state.last.error));
    }
    var total = open ? open.add + open.verify + open.relabel : 0;
    if (!r) {
      box.appendChild(el('p', 'muted', 'No check has run on this dataset yet. Start one above; its findings appear here.'));
    } else {
      var line = 'Last check: ' + when(r.created) + ', ' + r.folds_done + ' of ' + plural(r.folds, 'fold', 'folds') + ' done';
      if (r.cancelled) line += ', cancelled';
      line += '. ' + plural(r.frames_checked, 'frame', 'frames') + ' checked; ' + r.recipe.model + ' at ' +
        r.recipe.imgsz + ' px, ' + plural(r.recipe.epochs, 'epoch', 'epochs') + ' each.';
      box.appendChild(el('p', null, line));
    }
    if (r || total) {
      var wrap = el('div', 'scroll');
      var table = el('table', 'rcards');
      var head = el('thead');
      var hr = el('tr');
      ['Finding', 'Open', 'What it means', ''].forEach(function (h) { hr.appendChild(el('th', h === 'Open' ? 'num' : null, h)); });
      head.appendChild(hr);
      table.appendChild(head);
      var body = el('tbody');
      KINDS.forEach(function (k) {
        var tr = el('tr');
        tr.appendChild(el('td', 'rc-title', k.title));
        tr.appendChild(el('td', 'num', open ? String(open[k.key]) : '?')).setAttribute('data-label', 'Open');
        tr.appendChild(el('td', null, k.text)).setAttribute('data-label', 'What it means');
        var act = el('td', 'rc-action');
        act.appendChild(reviewLink(k.kind));
        tr.appendChild(act);
        body.appendChild(tr);
      });
      table.appendChild(body);
      wrap.appendChild(table);
      box.appendChild(wrap);
      box.appendChild(el('p', 'muted small', 'Open: suggestions nobody answered yet and that still fit the frame' +
        (open ? ' (' + plural(open.frames, 'frame', 'frames') + ').' : '.') +
        ' Nothing changes until you confirm one on the Review tab.'));
    }
    if (r) {
      box.appendChild(el('p', null, 'Loose labels counted: ' + r.loose));
      if (r.warnings && r.warnings.length) {
        var ul = el('ul', 'warn-list small');
        r.warnings.forEach(function (w) { ul.appendChild(el('li', null, w)); });
        box.appendChild(ul);
      }
    }
  }

  function recallText(x) {
    if (x.recall == null) return '-';
    return pct(x.recall) + ' (' + pct(x.low) + ' to ' + pct(x.high) + ')';
  }

  function recallTable(columns, rows) {
    var wrap = el('div', 'scroll');
    var table = el('table', 'rcards');
    var head = el('thead');
    var hr = el('tr');
    columns.forEach(function (c) { hr.appendChild(el('th', c.num ? 'num' : null, c.title)); });
    head.appendChild(hr);
    table.appendChild(head);
    var body = el('tbody');
    rows.forEach(function (row) {
      var tr = el('tr');
      columns.forEach(function (c, i) {
        var td = el('td', (i === 0 ? 'rc-title' : '') + (c.num ? ' num' : ''), c.value(row));
        if (i > 0) td.setAttribute('data-label', c.title);
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function drawCoverage(state) {
    var box = $('refine-coverage-body');
    clear(box);
    var r = state.report;
    if (!r) {
      box.appendChild(el('p', 'muted', 'No check has run yet. After one, this shows how often a model that never saw a camera day found the objects you labelled there.'));
      return;
    }
    box.appendChild(el('p', 'muted', 'Recall per camera, scene and class: how often a fold model that never saw that camera day found a labelled object (score of ' +
      num(r.found_score) + ' or more, overlapping the box by ' + num(r.found_iou) + ' or more), with a 95% interval. Few boxes give a wide interval.'));
    if (r.recall && r.recall.boxes) {
      box.appendChild(el('p', null, 'Overall: ' + recallText(r.recall) + ' of ' + plural(r.recall.boxes, 'box', 'boxes') + '.'));
    }
    var groups = r.recall_by_group || [];
    if (groups.length) {
      box.appendChild(recallTable([
        { title: 'Camera', value: function (g) { return g.camera; } },
        { title: 'Scene', value: function (g) { return g.scene; } },
        { title: 'Class', value: function (g) { return g.class; } },
        { title: 'Boxes', num: true, value: function (g) { return String(g.boxes); } },
        { title: 'Found', num: true, value: function (g) { return String(g.found); } },
        { title: 'Recall (95%)', num: true, value: recallText }
      ], groups));
    } else {
      box.appendChild(el('p', 'muted', 'The report holds no labelled boxes.'));
    }
    var sizes = r.recall_by_size || [];
    if (sizes.length) {
      box.appendChild(el('h3', null, 'By size'));
      box.appendChild(el('p', 'muted small', 'Size is the box\'s short side in pixels at the model\'s input. Objects below about 10 px are rarely found by any model, so a low recall there says little about your labels.'));
      box.appendChild(recallTable([
        { title: 'Size (px)', value: function (s) { return s.bucket; } },
        { title: 'Boxes', num: true, value: function (s) { return String(s.boxes); } },
        { title: 'Found', num: true, value: function (s) { return String(s.found); } },
        { title: 'Recall (95%)', num: true, value: recallText }
      ], sizes));
    }
  }

  function drawCapability(state) {
    var cap = state.capability || {};
    canCheck = cap.can_check !== false;
    var why = $('refine-why');
    why.hidden = canCheck;
    why.textContent = canCheck ? '' : (cap.why || 'This machine cannot train the fold models.');
    $('refine-start').disabled = !canCheck;
  }

  // ---- static objects

  var spotsById = {};
  var capDirty = false;
  var spotsSeq = 0;

  function reviewUrl(camera, extra) {
    return '/train/review?' + extra + (camera ? '&camera=' + encodeURIComponent(camera) : '');
  }

  // The place of a spot: the newest frame labelled there, cropped around the box (`crop`, all
  // normalised), the box drawn over it. Placed with style properties (the CSP has no inline style).
  function thumb(s) {
    if (!s.sample || !s.sample_size || !s.crop) return el('span', 'muted small spot-thumb-none', 'no frame to show');
    var c = s.crop, cw = c[2] - c[0], ch = c[3] - c[1];
    if (!(cw > 0 && ch > 0)) return el('span', 'muted small spot-thumb-none', 'no frame to show');
    var wrap = el('div', 'spot-thumb');
    wrap.style.aspectRatio = (cw * s.sample_size[0]) + ' / ' + (ch * s.sample_size[1]);
    var img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = 'A ' + s.label + ' parked on camera ' + s.camera;
    img.src = '/dataset/image?path=' + encodeURIComponent(s.sample);
    img.style.width = (100 / cw) + '%';
    img.style.height = (100 / ch) + '%';
    img.style.left = (-c[0] / cw * 100) + '%';
    img.style.top = (-c[1] / ch * 100) + '%';
    wrap.appendChild(img);
    var b = s.bbox, box = el('div', 'spot-box');
    box.style.left = ((b[0] - c[0]) / cw * 100) + '%';
    box.style.top = ((b[1] - c[1]) / ch * 100) + '%';
    box.style.width = ((b[2] - b[0]) / cw * 100) + '%';
    box.style.height = ((b[3] - b[1]) / ch * 100) + '%';
    wrap.appendChild(box);
    return wrap;
  }

  function drawSpots(j) {
    var box = $('refine-statics-body');
    clear(box);
    if (!capDirty) $('refine-cap').value = j.static_cap;
    $('refine-cap').max = j.static_cap_max;
    spotsById = {};
    if (j.folder_missing) {
      box.appendChild(el('p', 'error', j.message || 'The folder under review is not available.'));
      return;
    }
    var t = j.totals || {};
    if (!j.spots.length) {
      box.appendChild(el('p', 'muted', 'No static spots yet. A spot needs the same kind of object at the same place on at least 10 reviewed frames of one camera, from at least 2 days.'));
      return;
    }
    box.appendChild(el('p', null, plural(t.spots, 'static spot', 'static spots') + ' on ' +
      plural(t.labelled_boxes, 'labelled box', 'labelled boxes') + '; ' +
      plural(t.open_adds, 'missing box', 'missing boxes') + ' suggested there by the dataset check.' +
      (j.truncated ? ' The busiest ' + j.spots.length + ' are listed.' : '')));
    var wrap = el('div', 'scroll');
    var table = el('table', 'rcards');
    var head = el('thead');
    var hr = el('tr');
    ['Camera', 'What', 'Place', 'Frames', 'Days', 'Missing here', ''].forEach(function (h) {
      hr.appendChild(el('th', /^(Frames|Days|Missing here)$/.test(h) ? 'num' : null, h));
    });
    head.appendChild(hr);
    table.appendChild(head);
    var body = el('tbody');
    j.spots.forEach(function (s) {
      spotsById[s.id] = s;
      var tr = el('tr');
      tr.appendChild(el('td', 'rc-title', s.camera));
      tr.appendChild(el('td', null, s.label)).setAttribute('data-label', 'What');
      var place = el('td');
      place.setAttribute('data-label', 'Place');
      place.appendChild(thumb(s));
      tr.appendChild(place);
      tr.appendChild(el('td', 'num', String(s.labelled))).setAttribute('data-label', 'Frames');
      tr.appendChild(el('td', 'num', String(s.days))).setAttribute('data-label', 'Days');
      tr.appendChild(el('td', 'num', String(s.open_adds))).setAttribute('data-label', 'Missing here');
      var act = el('td', 'rc-action');
      var a = el('a', 'small', 'Review');
      a.href = reviewUrl(s.camera, 'auto=refine&refine_kind=add');
      act.appendChild(a);
      if (s.open_adds > 0) {
        act.appendChild(document.createTextNode(' '));
        act.appendChild(button('Accept the ' + s.open_adds + ' at this spot...', 'small secondary', function () { return askSpot(s.id); }));
      }
      tr.appendChild(act);
      body.appendChild(tr);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    box.appendChild(wrap);
    box.appendChild(el('p', 'muted small', 'Frames: reviewed frames with a labelled box at the spot. Missing here: suggestions of the dataset check for a box nobody drew at the spot; nothing is added until you confirm.'));
  }

  // The last read asked for wins (a read started before an accept may land after it).
  function loadSpots() {
    var seq = ++spotsSeq;
    return getJson('/v1/refine/spots')
      .then(function (j) { if (seq === spotsSeq) drawSpots(j); })
      .catch(function (e) {
        if (seq !== spotsSeq) return;
        var box = $('refine-statics-body');
        clear(box);
        box.appendChild(el('p', 'error', e.message));
      });
  }

  // A line under the card's table: the dry run's or the accept's answer, or why it failed.
  function spotSay(text, bad) {
    var box = $('refine-spot-result');
    clear(box);
    box.appendChild(el('p', bad ? 'error' : null, text));
    box.hidden = false;
    return box;
  }

  // Accepting the open adds of a spot: a dry run says what would happen, the box under the card
  // asks (inline, no dialog), the second step applies it. Both steps name the spot's id, never a
  // list of frames: the server judges every frame afresh.
  var pending = null;
  function hideSpotConfirm() {
    $('refine-spot-confirm').hidden = true;
    pending = null;
  }
  function spotName(s) { return 'a ' + s.label + ' on camera ' + s.camera; }

  function askSpot(id) {
    var s = spotsById[id];
    hideSpotConfirm();
    $('refine-spot-result').hidden = true;
    return postJson('/v1/refine/accept-spot', { spot: id, dry_run: true }).then(function (j) {
      var text;
      var weak = j.below_score > 0
        ? '; ' + j.below_score + ' weaker ' + (j.below_score === 1 ? 'one stays' : 'ones stay') + ' for one-by-one review'
        : '';
      var go = $('refine-spot-confirm-go');
      if (j.accepted === 0) {
        text = 'Nothing to add at this spot (' + spotName(s) + ') at a score of ' + num(j.min_score) + ' or more' + weak + '.';
        go.hidden = true;
      } else {
        var look = Math.min(10, j.frames);
        text = 'Add ' + plural(j.accepted, 'box', 'boxes') + ' on ' + plural(j.frames, 'frame', 'frames') +
          ' at this spot (' + spotName(s) + ', score ' + num(j.min_score) + ' or more' + weak + '). About 1 in 10 such suggestions is wrong: you will get ' +
          (j.frames > look ? look + ' of the frames' : 'the frames') + ' to look at afterwards.';
        if (j.eligible > j.frames) text += ' ' + (j.eligible - j.frames) + ' more frame(s) are beyond the limit of one run: run it again after.';
        go.hidden = false;
        go.textContent = 'Add them';
      }
      $('refine-spot-confirm-text').textContent = text;
      $('refine-spot-confirm').hidden = false;
      pending = { id: id, accepted: j.accepted };
      (go.hidden ? $('refine-spot-confirm-cancel') : go).focus();
    }).catch(function (e) { spotSay(e.message, true); });
  }

  function applySpot() {
    if (!pending) return;
    var go = $('refine-spot-confirm-go');
    var s = spotsById[pending.id];
    var id = pending.id;
    go.disabled = true;
    postJson('/v1/refine/accept-spot', { spot: id }).then(function (j) {
      hideSpotConfirm();
      var box = spotSay(j.message);
      if (j.sample && j.sample.length && s) {
        box.appendChild(el('p', null, 'Check these:'));
        var ul = el('ul', 'small spot-sample');
        j.sample.forEach(function (rel) {
          var li = el('li');
          var a = el('a', null, rel);
          a.href = reviewUrl(s.camera, 'status=any&source=all&frame=' + encodeURIComponent(rel));
          li.appendChild(a);
          ul.appendChild(li);
        });
        box.appendChild(ul);
      }
      if (j.failed && j.failed.length) {
        var bad = el('ul', 'warn-list small');
        j.failed.forEach(function (t) { bad.appendChild(el('li', null, t)); });
        box.appendChild(bad);
      }
      return load();
    }).catch(function (e) {
      hideSpotConfirm();
      spotSay(e.message, true);
      return loadSpots();
    }).then(function () { go.disabled = false; });
  }

  $('refine-spot-confirm-go').addEventListener('click', applySpot);
  $('refine-spot-confirm-cancel').addEventListener('click', function () {
    hideSpotConfirm();
    $('refine-spot-result').hidden = true;
  });

  // The training cap: saved as the config's own setting (0 = off).
  $('refine-cap').addEventListener('input', function () { capDirty = true; });
  $('refine-cap-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var msg = $('refine-cap-msg');
    var btn = $('refine-cap-save');
    var n = whole($('refine-cap').value);
    msg.textContent = '';
    msg.className = 'form-msg';
    if (n === null || !isFinite(n) || n < 0 || n !== Math.floor(n)) {
      msg.textContent = 'Not saved: give a whole number, 0 or more (0 turns the cap off).';
      msg.className = 'form-msg error';
      return;
    }
    btn.disabled = true;
    postJson('/v1/refine/static-cap', { cap: n })
      .then(function (j) {
        capDirty = false;
        msg.textContent = j.message;
      })
      .catch(function (e) { msg.textContent = e.message; msg.className = 'form-msg error'; })
      .then(function () { btn.disabled = false; });
  });

  // The state of the page: one read at load, another when a check starts or ends.
  function load() {
    return getJson('/v1/refine')
      .then(function (state) {
        drawCapability(state);
        drawFindings(state);
        drawCoverage(state);
        if (state.job) { drawJob(state.job); poller.kick(); } else if (jobId !== null) { clearJob(); }
        // The spots walk the reviewed frames: after the first paint, not before it.
        loadSpots();
      })
      .catch(function (e) {
        var box = $('refine-findings-body');
        clear(box);
        box.appendChild(el('p', 'error', e.message));
      });
  }
  load();
})();
