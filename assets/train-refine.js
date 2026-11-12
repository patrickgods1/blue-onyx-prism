// Refine tab (/train/refine): starts the dataset check and shows what it found. The Check card
// offers the full check (fold models) or a quick check with a model trained here (`check_mode`,
// the model select filled from GET /v1/refine `capability.models`; folds and epochs hidden and not
// sent then), previews (GET /v1/refine/preview) and starts (POST /v1/refine/check) a check and follows it
// through PrismPoll on GET /v1/train/jobs?tasks=busy (every 3 s while one is listed, every 30 s
// while none is, so a check started elsewhere shows up), with Pause / Resume / Cancel posted to the
// queue's routes. The Findings and Coverage cards are drawn from GET /v1/refine (each mode's last
// report: `report` of the full check, `quick_report` of the quick one, RefineFloor.reports; the
// open suggestions per kind) and read again when a check ends. The check changes no box, so
// nothing here confirms a box: Review does. The one question asked here is whether a start may
// replace the open suggestions of the frames it checks (inline, after the server's 409). The
// Static objects card (GET /v1/refine/spots) lists the places where one parked object sits across
// a camera's frames, with the one bulk action of this tab: "Accept the N at this spot" adds the
// check's missing boxes there (POST /v1/refine/accept-spot: a dry run first, then the inline
// question, then the write; the answer names frames to look at in Review) and the training cap
// (POST /v1/refine/static-cap). No inline handlers (CSP); limits come from the inputs' min/max and
// the server's answers.
// The Too small card (GET /v1/refine/floor, POST /v1/refine/size-floor, /ignore-mode and
// /resolve-small) shows what the size floor does per class, saves the floor and the ignore mode and
// has the one rule that writes boxes: pending proposals under the floor are hidden (dismissed,
// `reason: size_floor`), and those a lower floor or none no longer covers are restored (each a dry
// run, the inline question, then the write, as the Static objects card does). The last check's
// recall by size is drawn there too. The pure sentences are in `RefineFloor`, tested
// in Node (tests/it/js_rules.rs). No inline handlers (CSP); limits come from the server's answers.
// The Coverage card's second table, "Where the data is thin" (GET /v1/refine/coverage), is a row per
// camera x scene of the folder under review: reviewed frames, person / vehicle boxes, frames
// confirmed empty, the last check's recall there, unreviewed frames with a person (a link into Review
// under the page's own filters) and a short "what to do"; the rules and thresholds are in
// `RefineFloor` (node-tested).
(function () {
  'use strict';
  // The fewest boxes of a class a run wants to learn it from (the dataset check's MIN_CLASS_BOXES).
  var MIN_CLASS_BOXES = 50;
  function num(x) {
    var v = x == null ? NaN : Number(x);
    return isFinite(v) ? String(Number(v.toFixed(2))) : '?';
  }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }
  var api = {
    MIN_CLASS_BOXES: MIN_CLASS_BOXES,
    // A floor as a phrase: "under 10 px at 384".
    phrase: function (f) { return 'under ' + num(f.px) + ' px at ' + f.input; },
    // What a floor does to one class row ({class, boxes, under, left}), in words; '' when it
    // touches none of the class's boxes (then the floor is not to blame for a small class).
    classNote: function (c) {
      if (!c || !(c.under > 0)) return '';
      if (c.left === 0) {
        return 'every ' + c.class + ' label would be left out: the model cannot learn the class from these frames';
      }
      if (c.left < MIN_CLASS_BOXES) {
        return 'only ' + plural(c.left, c.class + ' label', c.class + ' labels') + ' would be left (under ' +
          MIN_CLASS_BOXES + '): too few to learn the class from';
      }
      return '';
    },
    // The question before the rule writes: `boxes` pending proposals on `frames` unreviewed frames.
    resolveText: function (boxes, frames) {
      return 'Hide ' + plural(boxes, 'pending auto-label proposal', 'pending auto-label proposals') + ' on ' +
        plural(frames, 'unreviewed frame', 'unreviewed frames') + ' as under the floor. ' +
        'They are set aside unseen like the auto-labelers\' hidden boxes: no label, no negative, not counted as your decisions. ' +
        'i on the Review tab shows them, and they can be restored here if the floor goes down or off.';
    },
    // How the last check's recall by size was measured (report.json): the fold models' labels and
    // the floor its judge used.
    measuredWith: function (r) {
      if (r && r.mode === 'model') {
        var m = r.model || {};
        var floorText = r.size_floor
          ? 'the check treated labels ' + api.phrase(r.size_floor) + ' as ignore regions (counted in this table only).'
          : 'no size floor was set.';
        return 'Measured by ' + (m.name || 'the model') + ' on ' + plural(r.frames_checked || 0, 'frame', 'frames') +
          ' it never saw, at its own input of ' + (m.imgsz || (r.recipe && r.recipe.imgsz) || '?') +
          ' px: what that model finds, not what the recipe can learn; ' + floorText;
      }
      if (!r || !r.folds_every_label) {
        return 'Measured by a check of an earlier version: if a size floor was saved then, its fold models never learned the objects under it, so their recall says little. Run the check again to measure with every label.';
      }
      var floor = r.size_floor
        ? 'the check treated labels ' + api.phrase(r.size_floor) + ' as ignore regions (counted in this table only).'
        : 'no size floor was set.';
      return 'Measured with fold models trained on every label; ' + floor;
    },
    // The question before a restore: `boxes` hidden proposals the saved floor no longer covers.
    restoreText: function (boxes, frames) {
      return 'Restore ' + plural(boxes, 'hidden proposal', 'hidden proposals') + ' on ' +
        plural(frames, 'unreviewed frame', 'unreviewed frames') + ' that the size floor no longer covers: ' +
        'they become pending again, for you to decide on the Review tab.';
    },
    // The Findings card's line on loose labels of report `r`: how many the last check counted and,
    // when it looked closer at them, how many got a proposed box (report.json `tighten`,
    // `found.tighten`).
    looseLine: function (r) {
      var n = (r && r.loose) || 0;
      if (!r || !r.tighten) return 'Loose labels counted: ' + n + ' (no closer look in this check).';
      var m = (r.found && r.found.tighten) || 0;
      return 'Loose labels counted: ' + n + ', ' + m + ' with a proposed box.';
    },
    // The Findings card's "Last check" line of report `r` (`when`: its time as the page prints
    // it): a full check's folds, or a quick check's model and the frames it skipped, so a quick
    // check never reads as a full one (report.json `mode`, `model`).
    lastLine: function (r, when) {
      var line = 'Last check: ' + when;
      if (r.mode === 'model') {
        var m = r.model || {};
        line += ', quick check with ' + (m.name || 'a model') + (m.run ? ' (training run ' + m.run + ')' : '');
        if (r.cancelled) line += ', cancelled';
        line += '. ' + plural(r.frames_checked || 0, 'frame', 'frames') + ' it never saw checked; ' +
          plural(m.trained_on || 0, 'frame', 'frames') + ' it trained on and ' + (m.validated_on || 0) +
          ' it validated on skipped (their suggestions stay as they were); ' + (r.recipe ? r.recipe.model : '?') +
          ' at ' + (m.imgsz || (r.recipe && r.recipe.imgsz) || '?') + ' px.';
        return line;
      }
      line += ', ' + r.folds_done + ' of ' + plural(r.folds, 'fold', 'folds') + ' done';
      if (r.cancelled) line += ', cancelled';
      line += '. ' + plural(r.frames_checked, 'frame', 'frames') + ' checked; ' + r.recipe.model + ' at ' +
        r.recipe.imgsz + ' px, ' + plural(r.recipe.epochs, 'epoch', 'epochs') + ' each.';
      return line;
    },
    // Who the Coverage card's recall is of: the fold models blind to a camera day, or a quick
    // check's model on the frames it never saw (with their count).
    recallOf: function (r) {
      if (r && r.mode === 'model') {
        var m = r.model || {};
        return 'Recall per camera, scene and class over the ' + plural(r.frames_checked || 0, 'frame', 'frames') + ' ' +
          (m.name || 'the model') + ' never saw: how often it found a labelled object';
      }
      return 'Recall per camera, scene and class: how often a fold model that never saw that camera day found a labelled object';
    },
    // The Coverage card's "Where the data is thin" table (GET /v1/refine/coverage).
    // Fewest person boxes of one camera and scene a model can be expected to learn people there
    // from: below this the class is a handful of boxes (the whole run wants MIN_CLASS_BOXES).
    MIN_PERSON_BOXES: 20,
    // Fewest person boxes in the last check's recall for a camera and scene before its interval is
    // trusted to say "missed": under this the 95% interval is too wide to blame the labels.
    MIN_RECALL_BOXES: 20,
    // A person recall whose interval ends under this (its upper bound) is missed for sure, not
    // by chance: even the best case is below 70%.
    MISSED_RECALL_HIGH: 0.7,
    // Share of a row's reviewed frames with a vehicle box (`vehicle_frames`, frames, never boxes)
    // from which "no empty frames" means a parked vehicle is in about every frame, not that none
    // were reviewed.
    VEHICLE_FRAME_SHARE: 0.9,
    // The Review link of a row's unreviewed person frames: the page's own filters (unreviewed,
    // all collected frames, the row's scene and camera, class person), which list the same
    // frames the row counts.
    coverageUrl: function (row) {
      return '/train/review?status=unreviewed&source=all&scene=' + encodeURIComponent(row.scene) +
        '&camera=' + encodeURIComponent(row.camera) + '&label=person';
    },
    // What to do about a row, in words ('' when there is nothing to say): few persons (review the
    // proposals there when there are some, else collect), no empty frame (reviewed, nothing in it; on a camera with
    // reviewed frames), persons missed (the recall interval's upper bound under MISSED_RECALL_HIGH
    // with MIN_RECALL_BOXES boxes behind it).
    coverageAdvice: function (row) {
      var out = [];
      var n = row.unreviewed_person_frames || 0;
      if ((row.persons || 0) < api.MIN_PERSON_BOXES) {
        out.push(n > 0
          ? 'Few persons: review the ' + plural(n, 'unreviewed frame', 'unreviewed frames') + ' with a person proposal'
          : 'Few persons: collect more here');
      }
      if ((row.reviewed || 0) > 0 && !(row.empty > 0)) {
        // A camera whose every frame shows a parked vehicle cannot have an empty frame: say so
        // instead of asking for the impossible. Frames with a vehicle against reviewed frames
        // (five cars on two frames are still two frames).
        out.push((row.vehicle_frames || 0) >= api.VEHICLE_FRAME_SHARE * row.reviewed
          ? 'No empty frames: about every frame here shows a vehicle, so false alerts on empty frames cannot be measured on this camera'
          : 'No empty frames: review some frames with nothing in them');
      }
      var p = row.person;
      if (p && p.boxes >= api.MIN_RECALL_BOXES && p.high < api.MISSED_RECALL_HIGH) {
        out.push('Persons are missed here: check the labels and collect more');
      }
      return out.length ? out.join('. ') + '.' : '';
    },
    // The Empty frames cell: reviewed frames with nothing in them, and how many the reviewer ticked.
    emptyText: function (row) {
      var n = row.empty || 0, c = row.empty_confirmed || 0;
      return c > 0 ? n + " (" + c + " confirmed)" : String(n);
    },
    // Who the table's recall columns are of (`r` = the report the recall comes from, or null): the
    // same words as the recall table above, so a quick check's recall reads as covering only the
    // frames its model never saw.
    thinRecallNote: function (r) {
      if (!r) return 'No check has run yet, so the recall columns show "-".';
      return api.recallOf(r) + '. Vehicles are car, truck, bus, van, motorcycle and bicycle together.';
    },
    // GET /v1/refine keeps each mode's last report: `report` (the full check's, report.json) and
    // `quick_report` (report-quick.json). `latest`: the newer of the two (the Findings card's);
    // `other`: the older one, when there is one; `recall`: the one the recall tables are drawn
    // from (the full check's when there is one: its whole-folder recall and recall by size are
    // what a floor is chosen from; else the quick check's).
    reports: function (state) {
      var full = (state && state.report) || null, quick = (state && state.quick_report) || null;
      var quickNewer = !!quick && (!full || String(quick.created || '') > String(full.created || ''));
      return {
        latest: quickNewer ? quick : full,
        other: quickNewer ? full : quick,
        recall: full || quick
      };
    },
    // The line under the Findings header naming the other mode's last report (`when`: its time as
    // the page prints it); '' without one.
    otherLine: function (other, when) {
      if (!other) return '';
      return (other.mode === 'model' ? 'Last quick check: ' : 'Last full check: ') + when +
        ' (its suggestions are counted above with the others).';
    }
  };
  globalThis.RefineFloor = api;
})();

(function () {
  'use strict';
  if (typeof document === 'undefined') return; // Node: the pure helpers above only
  var root = document.getElementById('refine-tab');
  if (!root) return;
  var $ = function (id) { return document.getElementById(id); };
  var form = $('refine-form');
  var KINDS = [
    { kind: 'add', title: 'Missing boxes', text: 'Objects a model that never saw the frame found that no box labels.', key: 'add' },
    { kind: 'verify', title: 'Labels no model found', text: 'Boxes the checking models never found: look at them, some are wrong.', key: 'verify' },
    { kind: 'relabel', title: 'Other class suggested', text: 'Boxes a checking model would give another class.', key: 'relabel' },
    { kind: 'tighten', title: 'Loose boxes', text: 'Labels that cover their object only partly: a checking model and a closer look by a larger model agree on a better-fitting box.', key: 'tighten' }
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

  // The check's mode: 'folds' (the full check) or 'model' (a quick check with a model trained
  // here, which trains nothing: no folds or epochs are sent with it).
  function mode() {
    var picked = form.querySelector('input[name="check_mode"]:checked');
    return picked ? picked.value : 'folds';
  }
  function values() {
    var f = form.elements;
    var quick = mode() === 'model';
    var o = {
      folds: quick ? null : whole(f.folds.value),
      epochs: quick ? null : whole(f.epochs.value),
      scenes: f.scene.value ? [f.scene.value] : [],
      cameras: list(f.camera.value),
      allow_large: !!f.allow_large.checked,
      // The box starts as the check's default (ticked when RF-DETR L's weights are here), so
      // sending it always says what the form shows.
      tighten: !!(f.tighten && f.tighten.checked)
    };
    if (quick) o.model = f.model.value || '';
    if (f.gpu && f.gpu.value) o.gpu = f.gpu.value;
    return o;
  }
  function query(o) {
    var p = [];
    var add = function (k, v) { p.push(encodeURIComponent(k) + '=' + encodeURIComponent(v)); };
    if (o.model != null) add('model', o.model);
    if (o.folds != null) add('folds', o.folds);
    if (o.epochs != null) add('epochs', o.epochs);
    if (o.scenes.length) add('scene', o.scenes.join(','));
    if (o.cameras.length) add('camera', o.cameras.join(','));
    if (o.gpu) add('gpu', o.gpu);
    if (o.allow_large) add('allow_large', '1');
    add('tighten', o.tighten ? '1' : '0');
    return p.join('&');
  }

  function drawPreview(p) {
    var box = $('refine-preview');
    clear(box);
    box.hidden = false;
    var m = p.mode === 'model' ? (p.model || {}) : null;
    if (m) {
      box.appendChild(el('p', null, 'Would check the ' + plural(p.frames, 'frame', 'frames') + ' ' + m.name +
        ' never saw, on ' + plural(p.camera_days, 'camera day', 'camera days') + ' (' + p.filter + ').'));
      if (m.run) {
        box.appendChild(el('p', null, 'Skipped: ' + plural(m.trained_on, 'frame', 'frames') + ' it trained on and ' +
          m.validated_on + ' it validated on (training run ' + m.run + ', ' + m.recipe + ' at ' + m.imgsz + ' px' +
          (m.device ? ', runs on ' + m.device : '') + '). Their suggestions stay as they are.'));
      }
    } else {
      box.appendChild(el('p', null, 'Would check ' + plural(p.frames, 'frame', 'frames') + ' on ' +
        plural(p.camera_days, 'camera day', 'camera days') + ' (' + p.filter + ').'));
    }
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
      // last_check = what the last quick check with the same model measured per frame.
      var basis = p.time_basis === 'last_run' ? 'from your last training run'
        : p.time_basis === 'last_check' ? 'from the last quick check with this model' : 'a rough guess';
      box.appendChild(el('p', null, m
        ? 'Time: ' + seconds(p.seconds) + ' (' + basis + '), on the model\'s own device.'
        : 'Time: ' + seconds(p.seconds) + ' on this GPU (' + basis + '). The GPU is busy meanwhile.'));
    }
    if (p.warnings && p.warnings.length) {
      var w = el('ul', 'warn-list small');
      p.warnings.forEach(function (t) { w.appendChild(el('li', null, t)); });
      box.appendChild(w);
    }
    if (p.refused) box.appendChild(el('p', 'error', 'A start would be refused: ' + p.refused));
    else box.appendChild(el('p', 'muted small', 'Nothing was run or written.'));
  }

  // A quick check without a model would be sent as a full one: refused here instead.
  function noModel() {
    if (mode() !== 'model' || form.elements.model.value) return false;
    say('Choose a model for the quick check (none trained here yet: run the full check).', true);
    return true;
  }

  $('refine-preview-btn').addEventListener('click', function () {
    var btn = $('refine-preview-btn');
    if (noModel()) return;
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
  // With the closer look, open better boxes (open_tightens) count too, as a bound: the look
  // replaces only those of the labels it answers.
  function askReplace(b) {
    $('refine-confirm-text').textContent = b.message;
    var tightens = b.open_tightens || 0;
    $('refine-confirm-start').textContent = 'Start and replace ' + (tightens ? 'up to ' : '') +
      plural((b.open_suggestions || 0) + tightens, 'open suggestion', 'open suggestions');
    say('');
    $('refine-confirm').hidden = false;
    $('refine-confirm-start').focus();
  }

  function start(replace) {
    var btn = $('refine-start');
    var again = $('refine-confirm-start');
    if (noModel()) return;
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
    var picked = RefineFloor.reports(state);
    var r = picked.latest, open = state.open;
    if (state.folder_missing) {
      box.appendChild(el('p', 'error', state.message || 'The folder under review is not available.'));
      return;
    }
    if (state.report_problem) box.appendChild(el('p', 'error', state.report_problem));
    if (state.quick_report_problem) box.appendChild(el('p', 'error', state.quick_report_problem));
    if (state.last && state.last.state === 'failed' && state.last.error && !state.job) {
      box.appendChild(el('p', 'error', 'The last check failed: ' + state.last.error));
    }
    var total = open ? open.add + open.verify + open.relabel + (open.tighten || 0) : 0;
    if (!r) {
      box.appendChild(el('p', 'muted', 'No check has run on this dataset yet. Start one above; its findings appear here.'));
    } else {
      box.appendChild(el('p', null, RefineFloor.lastLine(r, when(r.created))));
      if (picked.other) {
        box.appendChild(el('p', 'muted small', RefineFloor.otherLine(picked.other, when(picked.other.created))));
      }
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
        tr.appendChild(el('td', 'num', open ? String(open[k.key] || 0) : '?')).setAttribute('data-label', 'Open');
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
        ' Nothing changes until you confirm one on the Review tab. Counts refresh within a minute.'));
    }
    if (r) {
      box.appendChild(el('p', null, RefineFloor.looseLine(r)));
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
    var r = RefineFloor.reports(state).recall;
    lastReport = r;
    clear(box);
    if (!r) {
      box.appendChild(el('p', 'muted', 'No check has run yet. After one, this shows how often a model that never saw a camera day found the objects you labelled there.'));
      return;
    }
    box.appendChild(el('p', 'muted', RefineFloor.recallOf(r) + ' (score of ' +
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
    if ((r.recall_by_size || []).length) {
      box.appendChild(el('p', 'muted small', 'Recall by size is on the Too small card above.'));
    }
  }

  // "Where the data is thin" (GET /v1/refine/coverage): a camera x scene row per place of the folder
  // under review, with the counts, the recall there (the full check's report, else the quick
  // check's: RefineFloor.reports) and what to do (RefineFloor).
  // Everything from the server goes in through textContent; the one link is built from the row's
  // own names by RefineFloor.coverageUrl.
  var lastReport = null;
  function thinRecall(x) {
    if (!x || x.recall == null) return '-';
    return recallText(x) + ' of ' + plural(x.boxes, 'box', 'boxes');
  }
  function drawThin(j) {
    var box = $('refine-thin-body');
    clear(box);
    if (!j) return;
    if (j.folder_missing) {
      box.appendChild(el('p', 'error', j.message || 'The folder under review is not available.'));
      return;
    }
    var rows = j.rows || [];
    if (!rows.length) {
      box.appendChild(el('p', 'muted', 'No frames in the folder under review yet.'));
      return;
    }
    box.appendChild(el('p', 'muted small', RefineFloor.thinRecallNote(lastReport)));
    var wrap = el('div', 'scroll');
    var table = el('table', 'rcards');
    var head = el('thead');
    var hr = el('tr');
    var cols = [
      ['Camera', false], ['Scene', false], ['Reviewed frames', true], ['Person boxes', true], ['Vehicle boxes', true],
      ['Empty frames', true], ['Person recall (95%)', true], ['Vehicle recall (95%)', true],
      ['Unreviewed with a person', true], ['What to do', false]
    ];
    cols.forEach(function (c) { hr.appendChild(el('th', c[1] ? 'num' : null, c[0])); });
    head.appendChild(hr);
    table.appendChild(head);
    var body = el('tbody');
    rows.forEach(function (r) {
      var tr = el('tr');
      var cell = function (i, cls, text) {
        var td = el('td', cls, text);
        if (i > 0) td.setAttribute('data-label', cols[i][0]);
        tr.appendChild(td);
        return td;
      };
      cell(0, 'rc-title', r.camera);
      cell(1, null, r.scene);
      cell(2, 'num', String(r.reviewed));
      cell(3, 'num', String(r.persons));
      cell(4, 'num', String(r.vehicles));
      cell(5, 'num', RefineFloor.emptyText(r));
      cell(6, 'num', thinRecall(r.person));
      cell(7, 'num', thinRecall(r.vehicle));
      var un = cell(8, 'num', r.unreviewed_person_frames > 0 ? null : '0');
      if (r.unreviewed_person_frames > 0) {
        var a = el('a', null, String(r.unreviewed_person_frames));
        a.href = RefineFloor.coverageUrl(r);
        a.title = 'Open these frames in Review';
        un.appendChild(a);
      }
      cell(9, 'rc-block', RefineFloor.coverageAdvice(r));
      body.appendChild(tr);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    box.appendChild(wrap);
    box.appendChild(el('p', 'muted small', 'Boxes are the accepted labels of reviewed frames (unsure and too-small objects left out, hidden classes not counted). Unreviewed with a person: unreviewed frames with a person nobody rejected (a proposal, or the detection of a frame nobody opened yet); the number opens them in Review. Counts refresh within a minute.'));
  }
  function loadThin() {
    return getJson('/v1/refine/coverage')
      .then(drawThin)
      .catch(function (e) {
        var box = $('refine-thin-body');
        clear(box);
        box.appendChild(el('p', 'error', e.message));
      });
  }

  // The last check's recall by box size, on the Too small card: where models stop finding objects
  // is where a size floor belongs.
  function drawSizeRecall(state) {
    var box = $('refine-floor-recall');
    clear(box);
    var r = RefineFloor.reports(state).recall;
    var sizes = r ? (r.recall_by_size || []) : [];
    if (!sizes.length) {
      box.appendChild(el('p', 'muted small', 'After a dataset check, this shows how often a model that never saw the camera day found objects of each size.'));
      return;
    }
    box.appendChild(el('h3', null, 'Recall by size'));
    box.appendChild(el('p', 'muted small', 'Size is the box\'s short side in pixels at the model\'s input (' + r.recipe.imgsz +
      ' px in the last check). Objects below about 10 px are rarely found by any model, so a low recall there says little about your labels.'));
    box.appendChild(el('p', 'muted small', RF.measuredWith(r)));
    box.appendChild(recallTable([
      { title: 'Size (px)', value: function (s) { return s.bucket; } },
      { title: 'Boxes', num: true, value: function (s) { return String(s.boxes); } },
      { title: 'Found', num: true, value: function (s) { return String(s.found); } },
      { title: 'Recall (95%)', num: true, value: recallText }
    ], sizes));
  }

  // The models a quick check can use (GET /v1/refine `capability.models`), kept so a reload of
  // the state does not lose the choice.
  var quickModels = [];
  var fullCan = true;
  var fullWhy = '';

  // A preview or the replace question describes the model chosen when it was asked: when the
  // choice changes, for any reason, they go.
  function choiceChanged() {
    $('refine-preview').hidden = true;
    hideConfirm();
  }

  function fillModels(cap) {
    var select = $('refine-model');
    var keep = select.value;
    quickModels = cap.models || [];
    clear(select);
    quickModels.forEach(function (m) {
      var o = el('option', null, m.name + ' (' + m.recipe + ' at ' + m.imgsz + ' px, run ' + m.run + ')' +
        (m.enabled ? '' : ', disabled'));
      o.value = m.name;
      select.appendChild(o);
    });
    if (keep && quickModels.some(function (m) { return m.name === keep; })) select.value = keep;
    // The chosen model went (a reload of the list): the select now shows another one, or none.
    if (keep && select.value !== keep) choiceChanged();
    var note = $('refine-model-note');
    var bad = cap.unusable || [];
    var text = quickModels.length
      ? 'Models trained here (with their training record). Preview counts the frames each never saw.'
      : 'No configured model was trained here with its training record, so a quick check has nothing to use: run the full check, or train a model on the Train tab.';
    if (bad.length) {
      text += ' Not offered: ' + bad.map(function (b) { return b.name + ' (' + b.why + ')'; }).join('; ');
    }
    note.textContent = text;
  }

  // Folds and epochs are the full check's; the model select the quick check's. A full check needs
  // a GPU to train on, a quick check a model to run.
  function applyMode() {
    var quick = mode() === 'model';
    Array.prototype.forEach.call(form.querySelectorAll('[data-full-only]'), function (n) { n.hidden = quick; });
    $('refine-model-field').hidden = !quick;
    canCheck = quick ? quickModels.length > 0 : fullCan;
    var why = $('refine-why');
    var text = quick
      ? (quickModels.length ? '' : 'No model trained here to run a quick check with.')
      : (fullCan ? '' : fullWhy);
    why.hidden = !text;
    why.textContent = text;
    $('refine-start').disabled = !canCheck;
  }

  function drawCapability(state) {
    var cap = state.capability || {};
    fullCan = cap.can_check !== false;
    fullWhy = cap.why || 'This machine cannot train the fold models.';
    fillModels(cap);
    applyMode();
  }

  Array.prototype.forEach.call(form.querySelectorAll('input[name="check_mode"]'), function (r) {
    r.addEventListener('change', function () { applyMode(); choiceChanged(); });
  });
  $('refine-model').addEventListener('change', choiceChanged);

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
      // A frame a reviewer confirmed empty never gets a box in bulk.
      if (j.left_empty > 0) {
        weak += '; ' + plural(j.left_empty, 'frame', 'frames') + ' confirmed empty ' +
          (j.left_empty === 1 ? 'is' : 'are') + ' left for one-by-one review';
      }
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

  // ---- too small

  var RF = globalThis.RefineFloor;
  var floorDirty = false;
  var floorSeq = 0;
  var modeDirty = false;
  var savedFloor = null;
  var smallPending = null;

  function formMsg(id, text, bad) {
    var m = $(id);
    m.textContent = text || '';
    m.className = bad ? 'form-msg error' : 'form-msg';
  }

  // The floor the form holds, or null with the reason said (the same bounds the server checks).
  function formFloor() {
    var pxEl = $('refine-floor-px'), inEl = $('refine-floor-input');
    var px = whole(pxEl.value), input = whole(inEl.value);
    var pxMin = Number(pxEl.min), pxMax = Number(pxEl.max), inMin = Number(inEl.min), inMax = Number(inEl.max);
    if (px === null || !isFinite(px) || px < pxMin || px > pxMax) {
      formMsg('refine-floor-msg', 'Give the short side in pixels, from ' + pxMin + ' to ' + pxMax + '.', true);
      return null;
    }
    if (input === null || !isFinite(input) || input !== Math.floor(input) || input < inMin || input > inMax) {
      formMsg('refine-floor-msg', 'Give the model input as a whole number of pixels, from ' + inMin + ' to ' + inMax + '.', true);
      return null;
    }
    return { px: px, input: input };
  }

  // The per-class table of a floor answer: boxes, under the floor, left, and in words where the
  // floor would leave a class too few labels.
  function drawFloorCounts(j) {
    var box = $('refine-floor-counts');
    clear(box);
    var c = j.counts, f = j.shown;
    if (j.folder_missing) {
      box.appendChild(el('p', 'error', j.message || 'The folder under review is not available.'));
      box.hidden = false;
      return;
    }
    if (!c || !f) { box.hidden = true; return; }
    box.hidden = false;
    var under = c.per_class.reduce(function (n, x) { return n + x.under; }, 0);
    box.appendChild(el('p', null, 'A floor ' + RF.phrase(f) + ' leaves out ' + plural(under, 'label', 'labels') + ' on ' +
      plural(c.frames_with, 'frame', 'frames') + ' of the ' + plural(c.frames, 'reviewed frame', 'reviewed frames') +
      (c.frames_only ? '; ' + plural(c.frames_only, 'frame', 'frames') + ' would keep no label to train (a training run leaves ' +
        (c.frames_only === 1 ? 'it' : 'them') + ' out)' : '') + '.'));
    if (c.per_class.length) {
      var wrap = el('div', 'scroll');
      var table = el('table', 'rcards');
      var head = el('thead');
      var hr = el('tr');
      ['Class', 'Boxes', 'Under the floor', 'Left', ''].forEach(function (h, i) {
        hr.appendChild(el('th', i >= 1 && i <= 3 ? 'num' : null, h));
      });
      head.appendChild(hr);
      table.appendChild(head);
      var body = el('tbody');
      c.per_class.forEach(function (x) {
        var tr = el('tr');
        tr.appendChild(el('td', 'rc-title', x.class));
        tr.appendChild(el('td', 'num', String(x.boxes))).setAttribute('data-label', 'Boxes');
        tr.appendChild(el('td', 'num', String(x.under))).setAttribute('data-label', 'Under the floor');
        tr.appendChild(el('td', 'num', String(x.left))).setAttribute('data-label', 'Left');
        var note = RF.classNote(x);
        var td = el('td', note ? 'error small' : 'small', note);
        tr.appendChild(td);
        body.appendChild(tr);
      });
      table.appendChild(body);
      wrap.appendChild(table);
      box.appendChild(wrap);
    } else {
      box.appendChild(el('p', 'muted small', 'No reviewed frame holds a labelled box yet.'));
    }
    if (c.warnings && c.warnings.length) {
      var ul = el('ul', 'warn-list small');
      c.warnings.forEach(function (t) { ul.appendChild(el('li', null, t)); });
      box.appendChild(ul);
    }
    box.appendChild(el('p', 'muted small', 'Boxes are the accepted and added ones of reviewed frames. Nothing was written.'));
  }

  // The saved floor's answer: the form, the state line, the counts, the rule's button.
  function drawFloor(j) {
    savedFloor = j.saved;
    var shown = j.saved || j.suggested;
    var pxEl = $('refine-floor-px'), inEl = $('refine-floor-input');
    if (j.bounds) {
      pxEl.min = j.bounds.px_min; pxEl.max = j.bounds.px_max;
      inEl.min = j.bounds.input_min; inEl.max = j.bounds.input_max;
    }
    if (!floorDirty) { pxEl.value = num(shown.px); inEl.value = shown.input; }
    $('refine-floor-off').hidden = !j.saved;
    var state = $('refine-floor-state');
    if (j.saved) {
      state.textContent = 'Saved: objects ' + RF.phrase(j.saved) + ' are too small.';
    } else {
      var from = j.suggested.from === 'last_check' ? 'the last dataset check\'s input size'
        : (j.suggested.from === 'last_run' ? 'your last training run\'s input size' : 'a common input size');
      state.textContent = 'No floor is saved: nothing is left out for size. The form holds a suggestion (10 px, ' + from + ').';
    }
    drawFloorCounts(j);
    var open = j.saved && j.pending && j.pending.boxes > 0;
    $('refine-floor-pending').hidden = !open;
    if (open) {
      $('refine-floor-resolve').textContent = 'Hide ' + plural(j.pending.boxes, 'pending proposal', 'pending proposals') + ' under the floor...';
    }
    var back = j.restorable && j.restorable.boxes > 0;
    $('refine-floor-restorable').hidden = !back;
    if (back) {
      $('refine-floor-restore').textContent = 'Restore ' + plural(j.restorable.boxes, 'proposal', 'proposals') + ' the floor no longer covers...';
    }
    var mode = j.ignore_mode === 'drop-frame' ? 'drop-frame' : 'background';
    if (!modeDirty) {
      Array.prototype.forEach.call($('refine-ignore-form').elements.mode, function (r) { r.checked = r.value === mode; });
    }
  }

  // The last read asked for wins (a read started before a save may land after it).
  function loadFloor() {
    var seq = ++floorSeq;
    return getJson('/v1/refine/floor')
      .then(function (j) { if (seq === floorSeq) drawFloor(j); })
      .catch(function (e) {
        if (seq !== floorSeq) return;
        formMsg('refine-floor-msg', e.message, true);
        $('refine-floor-state').textContent = '';
      });
  }

  ['refine-floor-px', 'refine-floor-input'].forEach(function (id) {
    $(id).addEventListener('input', function () { floorDirty = true; });
  });

  $('refine-floor-preview').addEventListener('click', function () {
    var f = formFloor();
    if (!f) return;
    var btn = $('refine-floor-preview');
    btn.disabled = true;
    formMsg('refine-floor-msg', '');
    getJson('/v1/refine/floor?px=' + encodeURIComponent(f.px) + '&input=' + encodeURIComponent(f.input))
      .then(drawFloorCounts)
      .catch(function (e) { formMsg('refine-floor-msg', e.message, true); })
      .then(function () { btn.disabled = false; });
  });

  function saveFloor(body) {
    var btn = $('refine-floor-save'), off = $('refine-floor-off');
    btn.disabled = true;
    off.disabled = true;
    formMsg('refine-floor-msg', '');
    hideSmallConfirm();
    $('refine-small-result').hidden = true;
    return postJson('/v1/refine/size-floor', body)
      .then(function (j) {
        floorDirty = false;
        formMsg('refine-floor-msg', j.message);
        return loadFloor();
      })
      .catch(function (e) { formMsg('refine-floor-msg', e.message, true); })
      .then(function () { btn.disabled = false; off.disabled = false; });
  }
  $('refine-floor-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var f = formFloor();
    if (f) saveFloor(f);
  });
  $('refine-floor-off').addEventListener('click', function () { saveFloor({ off: true }); });

  // The rule: pending proposals under the saved floor become too-small ignore regions. A dry run
  // says what would happen, the box under the card asks (inline, no dialog), the second step writes.
  function hideSmallConfirm() {
    $('refine-small-confirm').hidden = true;
    smallPending = null;
  }
  function smallSay(text, bad) {
    var box = $('refine-small-result');
    clear(box);
    box.appendChild(el('p', bad ? 'error' : null, text));
    box.hidden = false;
    return box;
  }
  // `restore`: the dismissals the saved floor no longer covers, back to pending; else the hide.
  function askSmall(restore) {
    var btn = $(restore ? 'refine-floor-restore' : 'refine-floor-resolve');
    btn.disabled = true;
    hideSmallConfirm();
    $('refine-small-result').hidden = true;
    postJson('/v1/refine/resolve-small', { dry_run: true, restore: restore }).then(function (j) {
      var go = $('refine-small-confirm-go');
      var text;
      if (j.boxes === 0) {
        text = restore ? 'Nothing to restore: the floor still covers every hidden proposal.'
          : 'Nothing to hide: no pending proposal is under the floor any more.';
        go.hidden = true;
      } else {
        text = restore ? RF.restoreText(j.boxes, j.eligible) : RF.resolveText(j.boxes, j.eligible);
        if (j.eligible > 500) text += ' One run does at most 500 frames: run it again after.';
        go.hidden = false;
      }
      go.textContent = restore ? 'Restore them' : 'Hide them';
      $('refine-small-confirm-text').textContent = text;
      $('refine-small-confirm').hidden = false;
      smallPending = { boxes: j.boxes, restore: restore };
      (go.hidden ? $('refine-small-confirm-cancel') : go).focus();
    }).catch(function (e) { smallSay(e.message, true); })
      .then(function () { btn.disabled = false; });
  }
  $('refine-floor-resolve').addEventListener('click', function () { askSmall(false); });
  $('refine-floor-restore').addEventListener('click', function () { askSmall(true); });
  $('refine-small-confirm-go').addEventListener('click', function () {
    if (!smallPending) return;
    var go = $('refine-small-confirm-go');
    go.disabled = true;
    postJson('/v1/refine/resolve-small', { restore: smallPending.restore }).then(function (j) {
      hideSmallConfirm();
      var box = smallSay(j.message);
      if (j.failed && j.failed.length) {
        var bad = el('ul', 'warn-list small');
        j.failed.forEach(function (t) { bad.appendChild(el('li', null, t)); });
        box.appendChild(bad);
      }
      return loadFloor();
    }).catch(function (e) {
      hideSmallConfirm();
      smallSay(e.message, true);
      return loadFloor();
    }).then(function () { go.disabled = false; });
  });
  $('refine-small-confirm-cancel').addEventListener('click', function () {
    hideSmallConfirm();
    $('refine-small-result').hidden = true;
  });

  // The ignore mode: saved as the config's own setting, read by the next training run.
  $('refine-ignore-form').addEventListener('change', function () { modeDirty = true; });
  $('refine-ignore-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var btn = $('refine-ignore-save');
    var mode = $('refine-ignore-form').elements.mode.value;
    btn.disabled = true;
    formMsg('refine-ignore-msg', '');
    postJson('/v1/refine/ignore-mode', { mode: mode })
      .then(function (j) { modeDirty = false; formMsg('refine-ignore-msg', j.message); })
      .catch(function (e) { formMsg('refine-ignore-msg', e.message, true); })
      .then(function () { btn.disabled = false; });
  });

  // The state of the page: one read at load, another when a check starts or ends.
  function load() {
    return getJson('/v1/refine')
      .then(function (state) {
        drawCapability(state);
        drawFindings(state);
        drawCoverage(state);
        drawSizeRecall(state);
        if (state.job) { drawJob(state.job); poller.kick(); } else if (jobId !== null) { clearJob(); }
        // The spots walk the reviewed frames: after the first paint, not before it.
        loadSpots();
        loadFloor();
        loadThin();
      })
      .catch(function (e) {
        var box = $('refine-findings-body');
        clear(box);
        box.appendChild(el('p', 'error', e.message));
      });
  }
  load();
})();
