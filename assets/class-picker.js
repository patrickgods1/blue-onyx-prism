// Blue Onyx Prism: class picker widget (Review, Training and Config pages; served from /static/).
// ClassPicker.create(container, {entries, selected, mode, allowAdd, quickSets, onChange, extra})
// renders the classes as chips (checkbox + label) grouped People / Vehicles / Animals / Other,
// with a search box, free-text add and badges (RF-DETR + G-DINO / G-DINO only, the models that
// emit the class). `entries` come from GET /v1/review/class-catalog:
//   {name, category: person|vehicle|animal|object, in_list, hidden, coco, models, phrases}.
// mode 'compact' is a drop-down (<details>) with a summary line, for filters.
// `extra(entry, chip)` may add controls to a chip (the Review Classes card's "don't want").
// Pure helpers (validName, groupEntries, mergeEntries, defaultSelection, quickSetsFor,
// classChange, visibleClasses, summaryText, ...) live in ClassPicker.util and are tested by
// the page tests; no libraries, no inline handlers (CSP).
(function (root) {
  'use strict';

  var GROUPS = [
    ['person', 'People'], ['vehicle', 'Vehicles'], ['animal', 'Animals'], ['object', 'Other']
  ];
  var CAMERA_SET = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'dog', 'cat'];
  var NAME_MAX = 64;
  var seq = 0;

  // ---- pure helpers -------------------------------------------------------------------------

  // A class name as typed: trimmed, 1-64 characters, no control characters. null when invalid.
  function validName(s) {
    var t = String(s == null ? '' : s).trim();
    if (!t || t.length > NAME_MAX) return null;
    if (/[\u0000-\u001f\u007f-\u009f]/.test(t)) return null;
    return t;
  }

  // Why a typed name is refused, for the message line ('' when it is fine).
  function nameProblem(s) {
    var t = String(s == null ? '' : s).trim();
    if (!t) return 'Type a class name first.';
    if (t.length > NAME_MAX) return 'A class name has at most ' + NAME_MAX + ' characters.';
    if (validName(t) == null) return 'A class name cannot contain control characters.';
    return '';
  }

  function parseList(text) {
    var seen = {};
    return String(text || '').split(/[,\n]/).map(function (s) { return s.trim(); })
      .filter(function (s) { if (!s || seen[s]) return false; seen[s] = true; return true; });
  }

  function indexOfName(list, name) {
    var low = String(name).toLowerCase();
    for (var i = 0; i < list.length; i++) if (String(list[i]).toLowerCase() === low) return i;
    return -1;
  }

  function category(e) {
    return ['person', 'vehicle', 'animal'].indexOf(e && e.category) >= 0 ? e.category : 'object';
  }

  function groupEntries(entries) {
    var g = { person: [], vehicle: [], animal: [], object: [] };
    (entries || []).forEach(function (e) { g[category(e)].push(e); });
    return g;
  }

  // The catalog's entries plus an entry for every name in `extra` lists that has none (a class of
  // the list or of a model the catalog did not describe), so every selectable name has a chip.
  function mergeEntries(entries, names) {
    var out = (entries || []).slice();
    var have = {};
    out.forEach(function (e) { have[e.name] = true; });
    (names || []).forEach(function (n) {
      if (n && !have[n]) { have[n] = true; out.push({ name: n, category: 'object', in_list: false, coco: false, models: [] }); }
    });
    return out;
  }

  // Classes the auto-label form starts with: `relevant` minus `hidden`, in list order.
  function defaultSelection(cat) {
    var hidden = cat.hidden || [];
    var rel = cat.relevant || [];
    var classes = cat.classes || [];
    var ordered = classes.filter(function (c) { return rel.indexOf(c) >= 0; });
    rel.forEach(function (c) { if (ordered.indexOf(c) < 0) ordered.push(c); });
    return ordered.filter(function (c) { return hidden.indexOf(c) < 0; });
  }

  // The classes Review offers (pickers, shortcuts): the list minus the hidden ones.
  function visibleClasses(classes, hidden) {
    return (classes || []).filter(function (c) { return (hidden || []).indexOf(c) < 0; });
  }

  // Classes of `names` that are not in the class list (they need appending before use).
  function missingFrom(names, classes) {
    return (names || []).filter(function (n) { return (classes || []).indexOf(n) < 0; });
  }

  // How a new class list differs from the saved one: append-only keeps the old order as a prefix.
  function classChange(oldList, newList) {
    var removed = oldList.filter(function (c) { return newList.indexOf(c) < 0; });
    var kept = newList.filter(function (c) { return oldList.indexOf(c) >= 0; });
    var reordered = false;
    var oldKept = oldList.filter(function (c) { return newList.indexOf(c) >= 0; });
    for (var i = 0; i < kept.length; i++) if (kept[i] !== oldKept[i]) reordered = true;
    var appended = newList.filter(function (c) { return oldList.indexOf(c) < 0; });
    // New names in the middle also move later ids.
    var firstNew = newList.findIndex(function (c) { return oldList.indexOf(c) < 0; });
    var lastOld = -1;
    newList.forEach(function (c, i) { if (oldList.indexOf(c) >= 0) lastOld = i; });
    var inserted = firstNew >= 0 && firstNew < lastOld;
    return {
      removed: removed, appended: appended, reordered: reordered || inserted,
      idsChange: removed.length > 0 || reordered || inserted
    };
  }

  // "any" / "person, car" / "person, car +3": the one-line summary of a selection.
  function summaryText(names, emptyText) {
    if (!names || !names.length) return emptyText == null ? 'any' : emptyText;
    if (names.length <= 3) return names.join(', ');
    return names.slice(0, 3).join(', ') + ' +' + (names.length - 3);
  }

  // "3 of 80 · person, car, truck", "all", for a model's classes.
  function countText(names, total) {
    if (!names || !names.length) return 'all';
    var s = names.length + (total ? ' of ' + total : '') + ' · ';
    return s + summaryText(names, '');
  }

  // Quick sets of the auto-label forms: [{label, names}].
  function quickSetsFor(cat) {
    var classes = cat.classes || [];
    var hidden = cat.hidden || [];
    var entries = cat.entries || [];
    var cats = {};
    entries.forEach(function (e) { cats[e.name] = category(e); });
    var visible = visibleClasses(classes, hidden);
    var pv = visible.filter(function (c) { return cats[c] === 'person' || cats[c] === 'vehicle' || c === 'person'; });
    var animals = visible.filter(function (c) { return cats[c] === 'animal'; });
    var sets = [
      { label: 'All classes', names: visible },
      { label: 'Person + vehicles', names: pv },
      { label: 'Animals', names: animals }
    ];
    (cat.models || []).forEach(function (m) {
      if (!m.enabled || !m.classes || !m.classes.length) return;
      var names = visible.filter(function (c) { return m.classes.indexOf(c) >= 0; });
      sets.push({ label: 'What ' + m.name + ' detects', names: names });
    });
    return sets;
  }

  // Quick choices for one model's classes (Config page): [{label, names | global: true}].
  function modelQuickSets(cat, modelName) {
    var entries = cat.entries || [];
    var models = cat.models || [];
    var me = models.filter(function (m) { return m.name === modelName; })[0] || { classes: [] };
    var own = me.classes || [];
    var cats = {};
    entries.forEach(function (e) { cats[e.name] = category(e); });
    var camera = own.filter(function (c) { return CAMERA_SET.indexOf(c) >= 0 || cats[c] === 'animal'; });
    var sets = [
      { label: 'All', names: [] },
      { label: 'Camera set', names: camera }
    ];
    models.forEach(function (m) {
      if (m.name === modelName) return;
      var f = m.object_filter && m.object_filter.length ? m.object_filter : m.classes;
      sets.push({ label: 'Same as ' + m.name, names: (f || []).filter(function (c) { return own.indexOf(c) >= 0 || !own.length; }) });
    });
    sets.push({ label: 'Use global', global: true });
    return sets;
  }

  var util = {
    validName: validName, nameProblem: nameProblem, parseList: parseList, groupEntries: groupEntries,
    mergeEntries: mergeEntries, defaultSelection: defaultSelection, visibleClasses: visibleClasses,
    missingFrom: missingFrom, classChange: classChange, summaryText: summaryText, countText: countText,
    quickSetsFor: quickSetsFor, modelQuickSets: modelQuickSets, indexOfName: indexOfName, GROUPS: GROUPS
  };

  // ---- catalog and appending -----------------------------------------------------------------

  var catalogPromise = null;

  function api(url, opts) {
    return fetch(url, opts).then(function (r) {
      return r.json().catch(function () { return null; }).then(function (j) {
        if (!r.ok || !j || j.success === false) throw new Error((j && j.message) || ('HTTP ' + r.status));
        return j;
      });
    });
  }

  // GET /v1/review/class-catalog (cached for the page; `fresh` refetches).
  function fetchCatalog(fresh) {
    if (!catalogPromise || fresh) {
      catalogPromise = api('/v1/review/class-catalog', { headers: { Accept: 'application/json' } });
      catalogPromise.catch(function () { catalogPromise = null; });
    }
    return catalogPromise;
  }

  // Resolves when every name of `selected` is a class of the list: names outside it are
  // appended (never reordered) after the user confirms; cancel rejects with {cancelled: true}.
  // Resolves with the (possibly new) catalog.
  function ensureInList(selected, cat) {
    var missing = missingFrom(selected, cat.classes || []);
    if (!missing.length) return Promise.resolve(cat);
    var ok = root.confirm('Add ' + missing.join(', ') + ' to the class list (collect.classes)? ' +
      'They are appended after the existing classes, so existing class ids and saved labels do not change.');
    if (!ok) { var c = new Error('Cancelled.'); c.cancelled = true; return Promise.reject(c); }
    return api('/v1/review/classes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ classes: (cat.classes || []).concat(missing), hidden: cat.hidden || [] })
    }).then(function (j) {
      cat.classes = j.classes || (cat.classes || []).concat(missing);
      if (j.hidden) cat.hidden = j.hidden;
      return cat;
    });
  }

  // ---- widget --------------------------------------------------------------------------------

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function badgeFor(entry) {
    var b;
    if (entry.coco) {
      b = el('span', 'cp-badge cp-both', 'RF-DETR + G-DINO');
      b.title = 'Both labelers can propose this class.';
    } else {
      b = el('span', 'cp-badge cp-gdino', 'G-DINO only');
      b.title = 'G-DINO-only classes are always triaged review: RF-DETR cannot confirm them, so you decide every box.';
    }
    return b;
  }

  function create(container, opts) {
    opts = opts || {};
    var compact = opts.mode === 'compact';
    var id = 'cp' + (++seq);
    var entries = mergeEntries(opts.entries, opts.selected);
    var order = [];                  // selected names, in the order they were chosen / given
    (opts.selected || []).forEach(function (n) { if (order.indexOf(n) < 0) order.push(n); });
    var inputs = {};                 // name -> checkbox
    var chips = {};                  // name -> chip element
    var query = '';

    container.textContent = '';
    var rootEl = el(compact ? 'details' : 'div', 'cp ' + (compact ? 'cp-compact' : 'cp-full'));
    var summary = null;
    var body = rootEl;
    if (compact) {
      summary = el('summary', 'cp-summary');
      rootEl.appendChild(summary);
      body = el('div', 'cp-body');
      rootEl.appendChild(body);
    }
    var quick = el('div', 'cp-quick');
    var search = el('input', 'cp-search');
    search.type = 'search';
    search.placeholder = 'Search classes';
    search.setAttribute('aria-label', 'Search classes');
    search.autocomplete = 'off';
    var groupsEl = el('div', 'cp-groups');
    var msg = el('p', 'cp-msg form-msg');
    msg.setAttribute('role', 'status');
    body.appendChild(quick);
    body.appendChild(search);
    body.appendChild(groupsEl);
    var addInput = null;
    if (opts.allowAdd) {
      var addRow = el('div', 'cp-add');
      addInput = el('input', 'cp-add-input');
      addInput.type = 'text';
      addInput.maxLength = NAME_MAX;
      addInput.placeholder = 'Add a class (free text)';
      addInput.setAttribute('aria-label', 'Add a class by name');
      var addBtn = el('button', 'small secondary', 'Add');
      addBtn.type = 'button';
      addRow.appendChild(addInput);
      addRow.appendChild(addBtn);
      body.appendChild(addRow);
      var doAdd = function () {
        var problem = nameProblem(addInput.value);
        if (problem) { say(problem, true); return; }
        var name = validName(addInput.value);
        var known = indexOfName(entries.map(function (e) { return e.name; }), name);
        if (known >= 0) name = entries[known].name;
        else entries.push({ name: name, category: 'object', in_list: false, coco: false, models: [], custom: true });
        if (order.indexOf(name) < 0) order.push(name);
        addInput.value = '';
        render();
        say(known >= 0 ? name + ' is already listed; selected.' : 'Added ' + name + ' (G-DINO only).', false);
        changed();
      };
      addBtn.addEventListener('click', doAdd);
      addInput.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter') { ev.preventDefault(); doAdd(); }
      });
    }
    body.appendChild(msg);
    container.appendChild(rootEl);

    function say(text, isError) {
      msg.textContent = text || '';
      msg.className = 'cp-msg form-msg' + (isError ? ' error-text' : '');
    }

    function selected() { return order.slice(); }

    function updateSummary() {
      if (summary) summary.textContent = (opts.label ? opts.label + ': ' : '') + summaryText(order, opts.emptyText);
    }

    function changed() {
      updateSummary();
      if (opts.onChange) opts.onChange(selected());
    }

    function applySearch() {
      var q = query.toLowerCase();
      Array.prototype.forEach.call(groupsEl.querySelectorAll('[data-group]'), function (fs) {
        var any = false;
        Array.prototype.forEach.call(fs.querySelectorAll('.cp-chip'), function (chip) {
          var hit = !q || chip.getAttribute('data-name').toLowerCase().indexOf(q) >= 0;
          chip.hidden = !hit;
          if (hit) any = true;
        });
        fs.hidden = !any;
      });
      // The classes outside the list: open while a search finds one of them.
      var more = groupsEl.querySelector('.cp-more');
      if (more) {
        var found = more.querySelector('.cp-chip:not([hidden])');
        more.hidden = !!q && !found;
        if (q && found) more.open = true;
      }
    }

    function chipFor(entry, inMore) {
      var inputId = id + '-' + Object.keys(inputs).length;
      var chip = el('span', 'cp-chip' + (entry.hidden ? ' cp-hidden' : '') + (entry.in_list === false ? ' cp-new' : ''));
      chip.setAttribute('data-name', entry.name);
      var cb = el('input');
      cb.type = 'checkbox';
      cb.id = inputId;
      cb.value = entry.name;
      cb.checked = order.indexOf(entry.name) >= 0;
      cb.addEventListener('change', function () {
        var i = order.indexOf(entry.name);
        if (cb.checked && i < 0) order.push(entry.name);
        if (!cb.checked && i >= 0) order.splice(i, 1);
        say('', false);
        changed();
      });
      var label = el('label', 'cp-label');
      label.setAttribute('for', inputId);
      label.appendChild(el('span', 'cp-name', entry.name));
      chip.appendChild(cb);
      chip.appendChild(label);
      if (!compact) {
        label.appendChild(badgeFor(entry));
        if (entry.models && entry.models.length) {
          var m = el('span', 'cp-models', entry.models.join(', '));
          m.title = 'Detected by: ' + entry.models.join(', ');
          label.appendChild(m);
        }
        if (entry.in_list === false && !inMore) {
          var n = el('span', 'cp-badge cp-notlist', 'not in list');
          n.title = 'Not in the class list (collect.classes) yet: using it asks to append it.';
          label.appendChild(n);
        }
      } else if (entry.models && entry.models.length) {
        label.title = 'Detected by: ' + entry.models.join(', ');
      }
      if (opts.extra) opts.extra(entry, chip);
      inputs[entry.name] = cb;
      chips[entry.name] = chip;
      return chip;
    }

    // The category fieldsets of `list` into `parent` (`key` prefixes their data-group).
    function groupsInto(parent, list, key, inMore) {
      var grouped = groupEntries(list);
      GROUPS.forEach(function (g) {
        var items = grouped[g[0]];
        if (!items.length) return;
        var fs = el('fieldset', 'cp-group');
        fs.setAttribute('data-group', key + g[0]);
        fs.appendChild(el('legend', null, g[1]));
        var wrap = el('div', 'cp-chips');
        items.forEach(function (e) { wrap.appendChild(chipFor(e, inMore)); });
        fs.appendChild(wrap);
        parent.appendChild(fs);
      });
    }

    function render() {
      groupsEl.textContent = '';
      inputs = {}; chips = {};
      // The full picker shows the class list first; the classes outside it (the labelers' whole
      // vocabulary, 100 or so) wait in a closed section that opens for a selected one.
      var outside = compact ? [] : entries.filter(function (e) { return e.in_list === false; });
      groupsInto(groupsEl, compact ? entries : entries.filter(function (e) { return e.in_list !== false; }), '', false);
      if (outside.length) {
        var more = el('details', 'cp-more');
        var sum = el('summary', null, 'More classes, not in your class list (' + outside.length + ')');
        sum.title = 'Not in the class list (collect.classes) yet: using one asks to append it.';
        more.appendChild(sum);
        groupsInto(more, outside, 'more-', true);
        more.open = outside.some(function (e) { return order.indexOf(e.name) >= 0; });
        groupsEl.appendChild(more);
      }
      if (!entries.length) groupsEl.appendChild(el('p', 'muted', 'No classes known yet.' + (opts.allowAdd ? ' Add one below.' : '')));
      applySearch();
      updateSummary();
    }

    function renderQuick() {
      quick.textContent = '';
      (opts.quickSets || []).forEach(function (q) {
        var b = el('button', 'small secondary', q.label);
        b.type = 'button';
        b.addEventListener('click', function () {
          if (opts.onQuick && opts.onQuick(q, api_)) return;
          api_.setSelected(q.names || []);
        });
        quick.appendChild(b);
      });
      quick.hidden = !(opts.quickSets && opts.quickSets.length);
    }

    search.addEventListener('input', function () { query = search.value.trim(); applySearch(); });
    search.addEventListener('keydown', function (ev) { if (ev.key === 'Enter') ev.preventDefault(); });

    var api_ = {
      el: rootEl,
      getSelected: selected,
      setSelected: function (names, silent) {
        order = [];
        (names || []).forEach(function (n) { if (order.indexOf(n) < 0) order.push(n); });
        var add = order.filter(function (n) { return !inputs[n]; });
        if (add.length) entries = mergeEntries(entries, add);
        render();
        if (!silent) changed();
      },
      setEntries: function (list) {
        entries = mergeEntries(list, order);
        render();
      },
      setQuickSets: function (q) { opts.quickSets = q; renderQuick(); },
      say: say,
      checkbox: function (name) { return inputs[name] || null; },
      chip: function (name) { return chips[name] || null; }
    };
    render();
    renderQuick();
    return api_;
  }

  root.ClassPicker = {
    create: create, fetchCatalog: fetchCatalog, ensureInList: ensureInList, util: util
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.ClassPicker;
})(typeof window !== 'undefined' ? window : globalThis);
