// The export card's line about the dataset check's open suggestions (GET /v1/refine `open`, the
// counts that route already keeps for 60 s; nothing is walked here): pure, node-tested
// (tests/it/js_rules.rs), `globalThis.DatasetOpen`.
(function () {
  'use strict';
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }
  globalThis.DatasetOpen = {
    // `open` = {add, verify, relabel, tighten, frames}: the sentence in three parts (before the
    // link, the link's text, after it), or null when nothing is open.
    line: function (open) {
      if (!open) return null;
      var a = open.add || 0, v = open.verify || 0, r = open.relabel || 0, t = open.tighten || 0;
      var total = a + v + r + t;
      if (!(total > 0)) return null;
      return {
        before: 'The dataset check has ' + plural(total, 'open suggestion', 'open suggestions') + ' (' +
          a + ' missing boxes, ' + v + ' labels to verify, ' + r + ' other class, ' + t + ' better boxes) on ' +
          plural(open.frames || 0, 'frame', 'frames') +
          '. An export uses the labels as they are: answer them on the ',
        link: 'Refine tab',
        after: ' first, or export anyway.'
      };
    }
  };
})();

// Dataset tab (/train/dataset): the Export form (POST /v1/review/export), the X-AnyLabeling import
// (POST /v1/review/import) and the Classes card (POST /v1/review/classes, the class picker of
// class-picker.js: the list in order, "hide" (don't want: hide + reject) per class =
// collect.hidden_classes). A class save that leaves reviewed frames with accepted boxes of unwanted
// classes asks, then POST /v1/review/reject-unwanted-reviewed. After an export the Exports and
// coverage cards are read again from the page's own URL. The export card warns when the dataset check left
// open suggestions (`DatasetOpen`, GET /v1/refine). No inline handlers (CSP); server values
// come from data-*.
(function () {
  'use strict';
  if (typeof document === 'undefined') return; // Node: the pure helper above only
  var root = document.getElementById('dataset-tab');
  if (!root) return;
  var classes = [];
  try { classes = JSON.parse(root.getAttribute('data-classes') || '[]'); } catch (e) { classes = []; }
  var CP = window.ClassPicker || null;
  var CU = CP ? CP.util : null;
  var hiddenClasses = [];
  var catalog = null;
  var classesPicker = null;
  var hideSet = {};   // class -> "don't want" ticked
  var $ = function (id) { return document.getElementById(id); };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
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

  function postJson(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body)
    }).then(function (r) {
      return r.json().catch(function () { return null; }).then(function (j) {
        if (!r.ok || !j || j.success === false) throw new Error((j && j.message) || ('HTTP ' + r.status));
        return j;
      });
    });
  }

  function list(text) {
    return String(text || '').split(/[,\n]/).map(function (s) { return s.trim(); }).filter(Boolean);
  }

  // The status strip's counts changed.
  function counted() { window.dispatchEvent(new Event('train-area:refresh')); }

  // The Exports and coverage cards as the server sees them now.
  function refreshCards() {
    return fetch(location.pathname, { headers: { Accept: 'text/html' } })
      .then(function (r) { return r.ok ? r.text() : null; })
      .then(function (html) {
        if (!html) return;
        var page = new DOMParser().parseFromString(html, 'text/html');
        ['exports', 'coverage'].forEach(function (id) {
          var now = page.getElementById(id), here = $(id);
          if (now && here) here.replaceWith(document.importNode(now, true));
        });
      })
      .catch(function () { /* the next load shows them */ });
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
        'Boxes per class: ' + (per || 'none') + '; ' + r.hard_negative_boxes + ' rejected box(es) in hard_negatives.json; ' +
          (r.ignore_boxes || 0) + ' unsure / too-small box(es) in ignore.json'
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
      counted();
      return refreshCards();
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
      counted();
      return refreshCards();
    }).catch(function (e) { report(box, [e.message]); });
  });

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
  // boxes of unwanted classes on reviewed frames too and appends the outcome to `box`.
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
    }).catch(function (e) {
      box.appendChild(el('div', 'error-text', e.message));
    });
  }

  function entries() {
    if (!CU) return [];
    var base = catalog && catalog.entries ? catalog.entries : [];
    var out = CU.mergeEntries(base, classes);
    out.forEach(function (e) {
      if (classes.indexOf(e.name) >= 0) e.in_list = true;
      e.hidden = hiddenClasses.indexOf(e.name) >= 0;
    });
    return out;
  }

  // Without the picker the comma text field saves as before.
  $('rv-classes').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var box = $('rv-classes-msg');
    var body;
    if (classesPicker) {
      var picked = classesPicker.getSelected();
      if (!picked.length) {
        box.textContent = 'Keep at least one class in the list.';
        box.className = 'form-msg error-text';
        return;
      }
      var change = CU.classChange(classes, picked);
      if (change.idsChange && !window.confirm(
        (change.removed.length ? 'Remove ' + change.removed.join(', ') + ' from the class list. ' : '') +
        (change.reordered ? 'The order of the classes changes. ' : '') +
        'YOLO class ids of later exports change (frames you exported before keep the old ids). Pending boxes of a ' +
        'removed class are rejected on the unreviewed frames; for reviewed frames you are asked next. Continue?')) {
        box.textContent = 'Not saved.';
        box.className = 'form-msg';
        return;
      }
      body = { classes: picked, hidden: picked.filter(function (c) { return hideSet[c]; }) };
    } else {
      body = { classes: list(ev.target.elements.classes.value) };
    }
    postJson('/v1/review/classes', body).then(function (j) {
      classes = j.classes;
      if (j.hidden) hiddenClasses = j.hidden;
      if (catalog) { catalog.classes = classes.slice(); catalog.hidden = hiddenClasses.slice(); }
      hideSet = {};
      classes.forEach(function (c) { hideSet[c] = hiddenClasses.indexOf(c) >= 0; });
      if (classesPicker) classesPicker.setEntries(entries());
      var lines = [j.message || 'Saved.'];
      var auto = unwantedNote(j);
      if (auto) lines.push(auto);
      box.className = 'form-msg ok-text';
      report(box, lines);
      return askReviewedUnwanted(j.reviewed_with_unwanted, box).then(counted);
    }).catch(function (e) { box.textContent = e.message; box.className = 'form-msg error-text'; });
  });

  function initPicker() {
    var holder = $('rv-classes-picker');
    if (!holder || !CP) return;
    classes.forEach(function (c) { hideSet[c] = hiddenClasses.indexOf(c) >= 0; });
    classesPicker = CP.create(holder, {
      entries: entries(), selected: classes.slice(), mode: 'full', allowAdd: true,
      extra: function (entry, chip) {
        var lab = el('label', 'cp-extra');
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = !!hideSet[entry.name];
        cb.setAttribute('aria-label', 'Do not want ' + entry.name + ': hide it in Review and reject its pending boxes');
        cb.addEventListener('change', function () { hideSet[entry.name] = cb.checked; });
        lab.appendChild(cb);
        lab.title = "Don't want " + entry.name + ": hide it in Review and reject its pending boxes";
        lab.appendChild(document.createTextNode('hide'));
        chip.appendChild(lab);
      }
    });
    var plain = $('rv-classes').elements.classes;
    if (plain) plain.hidden = true;
  }

  if (CP) {
    CP.fetchCatalog().then(function (c) {
      catalog = c;
      if (Array.isArray(c.classes)) classes = c.classes;
      hiddenClasses = Array.isArray(c.hidden) ? c.hidden : [];
    }).catch(function () { catalog = null; }).then(initPicker);
  }

  // The dataset check's open suggestions, before the Export button: one read of GET /v1/refine (its
  // counts are kept 60 s there, so this walks nothing new). Nothing is shown for none, an
  // unavailable folder or a failed read: the line only warns.
  (function () {
    var line = $('export-open');
    if (!line || !window.DatasetOpen) return;
    fetch('/v1/refine', { headers: { Accept: 'application/json' }, cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || j.success === false || j.folder_missing) return;
        var parts = window.DatasetOpen.line(j.open);
        if (!parts) return;
        line.textContent = parts.before;
        var a = el('a', null, parts.link);
        a.href = '/train/refine';
        line.appendChild(a);
        line.appendChild(document.createTextNode(parts.after));
        line.hidden = false;
      })
      .catch(function () { /* the line only warns */ });
  })();
})();
