// Blue Onyx Prism: model picker widget for the auto-label forms (Review and Training pages;
// served from /static/). ModelPicker.create(container, {onChange, selectName, deviceName})
// fetches GET /v1/autolabel/models and renders a drop-down grouped "In config" / "Model files" /
// "Downloadable / exportable" (name, family, class count, COCO, size, notes), a detail line, a
// device select (first option: the model's own device) and, when the chosen model needs them,
// the licence and large-download confirmations (named confirm_license / confirm_large).
// getSelection() -> {id, device, model}; id and device post as `with_model` and `device`.
// Pure helpers (groupModels, optionText, notesFor, ...) live in ModelPicker.util and are tested
// under node; no libraries, no inline handlers (CSP).
(function (root) {
  'use strict';

  var GROUPS = [
    ['config', 'In config'], ['local', 'Model files'], ['catalog', 'Downloadable / exportable']
  ];

  // ---- pure helpers -------------------------------------------------------------------------

  function groupOf(m) {
    return m && (m.source === 'config' || m.source === 'local') ? m.source : 'catalog';
  }

  // {config: [], local: [], catalog: []}, each in the order the server sent.
  function groupModels(models) {
    var g = { config: [], local: [], catalog: [] };
    (models || []).forEach(function (m) { g[groupOf(m)].push(m); });
    return g;
  }

  // "135.8 MB", "1.2 GB", "" when unknown.
  function sizeText(bytes) {
    var n = Number(bytes);
    if (!isFinite(n) || n <= 0) return '';
    if (n >= 1e9) return (n / 1e9).toFixed(1) + ' GB';
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + ' MB';
    return Math.max(1, Math.round(n / 1e3)) + ' KB';
  }

  function isCatalogOnly(m) { return !!m && !m.installed && groupOf(m) === 'catalog'; }

  // The server decides (`confirm_license`: a catalog model under a non-commercial / personal-use
  // licence); the page asks exactly then, so a run is never refused for a box it did not show.
  function needsLicenseConfirm(m) { return !!m && !!m.confirm_license; }

  // A large download is only confirmed for a model that is not installed yet.
  function needsLargeConfirm(m) { return !!m && !!m.large && !m.installed; }

  // Short flags of a row: ["needs export", "large download", "installs on first run", ...].
  function notesFor(m) {
    var out = [];
    if (!m) return out;
    if (m.needs_export) out.push('needs export');
    if (m.large && !m.installed) out.push('large download');
    if (isCatalogOnly(m)) out.push('installs on first run');
    if (needsLicenseConfirm(m)) out.push(m.license ? m.license + ', confirm below' : 'licence: confirm below');
    else if (m.license) out.push(m.license);
    return out;
  }

  // One line for the <option>: "rfdetr-large - rfdetr - 80 classes - COCO - 129.5 MB - needs export".
  function optionText(m) {
    var parts = [m.name || m.id];
    if (m.family) parts.push(m.family);
    if (m.classes != null && m.classes !== '') parts.push(m.classes + ' classes');
    if (m.coco) parts.push('COCO');
    var s = sizeText(m.size_bytes);
    if (s) parts.push(s);
    if (m.in_config && !m.enabled && groupOf(m) === 'config') parts.push('disabled');
    notesFor(m).forEach(function (n) { if (/^(needs export|large download|installs)/.test(n)) parts.push(n); });
    return parts.join(' · ');
  }

  // The detail text under the select.
  function detailText(m) {
    if (!m) return 'Choose the model that proposes the boxes.';
    var bits = [];
    if (m.coco) bits.push('COCO classes (80 or 91)');
    else if (m.classes) bits.push(m.classes + ' classes (a custom list)');
    if (m.note) bits.push(m.note);
    var notes = notesFor(m);
    if (notes.length) bits.push(notes.join(', '));
    if (m.device) bits.push('runs on ' + m.device + ' by default');
    return bits.join('. ');
  }

  // Why the pre-accept score is refused, '' when fine (empty = off).
  function acceptProblem(threshold, accept) {
    var a = String(accept == null ? '' : accept).trim();
    if (!a) return '';
    var av = Number(a);
    if (!isFinite(av) || av <= 0 || av > 1) return 'Pre-accept score must be between 0 and 1 (or empty for off).';
    var tv = Number(String(threshold == null ? '' : threshold).trim());
    if (isFinite(tv) && tv > 0 && av < tv) return 'Pre-accept score must be at least the threshold (' + tv + ').';
    return '';
  }

  // Why the selection cannot be submitted, '' when it can.
  function selectionProblem(m, confirms) {
    if (!m) return 'Choose a model first.';
    confirms = confirms || {};
    if (needsLicenseConfirm(m) && !confirms.license) return 'Confirm the licence of ' + (m.name || m.id) + ' first.';
    if (needsLargeConfirm(m) && !confirms.large) return 'Confirm the large download of ' + (m.name || m.id) + ' first.';
    return '';
  }

  // Label of the device select's first option.
  function ownDeviceLabel(m) { return "model's own (" + ((m && m.device) || 'auto') + ')'; }

  var util = {
    groupModels: groupModels, sizeText: sizeText, needsLicenseConfirm: needsLicenseConfirm,
    needsLargeConfirm: needsLargeConfirm, notesFor: notesFor, optionText: optionText,
    detailText: detailText, acceptProblem: acceptProblem, selectionProblem: selectionProblem,
    ownDeviceLabel: ownDeviceLabel, isCatalogOnly: isCatalogOnly, GROUPS: GROUPS
  };

  // ---- fetch ---------------------------------------------------------------------------------

  function fetchModels() {
    return fetch('/v1/autolabel/models', { headers: { Accept: 'application/json' } }).then(function (r) {
      return r.json().catch(function () { return null; }).then(function (j) {
        if (!r.ok || !j || j.success === false) throw new Error((j && j.message) || ('HTTP ' + r.status));
        return j;
      });
    });
  }

  // ---- widget --------------------------------------------------------------------------------

  var seq = 0;

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function create(container, opts) {
    opts = opts || {};
    var id = 'mp' + (++seq);
    var models = [];
    var data = { devices: [], default_threshold: 0.3 };
    container.textContent = '';
    var rootEl = el('div', 'mp');
    var selLabel = el('label', 'mp-label', 'Model');
    selLabel.setAttribute('for', id + '-model');
    var select = el('select', 'mp-select');
    select.id = id + '-model';
    select.name = opts.selectName || 'with_model';
    var detail = el('p', 'muted small mp-detail');
    var devLabel = el('label', 'mp-label', 'Device');
    devLabel.setAttribute('for', id + '-device');
    var device = el('select', 'mp-device');
    device.id = id + '-device';
    device.name = opts.deviceName || 'device';
    var licLabel = el('label', 'check mp-confirm');
    var lic = el('input');
    lic.type = 'checkbox'; lic.name = 'confirm_license';
    var licText = el('span', null, '');
    licLabel.appendChild(lic); licLabel.appendChild(licText);
    var bigLabel = el('label', 'check mp-confirm');
    var big = el('input');
    big.type = 'checkbox'; big.name = 'confirm_large';
    var bigText = el('span', null, '');
    bigLabel.appendChild(big); bigLabel.appendChild(bigText);
    var msg = el('p', 'mp-msg form-msg');
    msg.setAttribute('role', 'status');
    [selLabel, select, detail, devLabel, device, licLabel, bigLabel, msg].forEach(function (n) { rootEl.appendChild(n); });
    container.appendChild(rootEl);

    function current() {
      for (var i = 0; i < models.length; i++) if (models[i].id === select.value) return models[i];
      return null;
    }

    function say(text, isError) {
      msg.textContent = text || '';
      msg.className = 'mp-msg form-msg' + (isError ? ' error-text' : '');
    }

    function renderDevices(keep) {
      var m = current();
      device.textContent = '';
      var own = el('option', null, ownDeviceLabel(m));
      own.value = '';
      device.appendChild(own);
      (data.devices || []).forEach(function (d) {
        var o = el('option', null, d.label || d.value);
        o.value = d.value;
        device.appendChild(o);
      });
      device.value = keep || '';
      if (device.value !== (keep || '')) device.value = '';
    }

    function sync() {
      var m = current();
      detail.textContent = detailText(m);
      var needLic = needsLicenseConfirm(m);
      var needBig = needsLargeConfirm(m);
      licLabel.hidden = !needLic;
      bigLabel.hidden = !needBig;
      if (!needLic) lic.checked = false;
      if (!needBig) big.checked = false;
      licText.textContent = needLic
        ? ' I accept the licence of ' + (m.name || m.id) + (m.license ? ' (' + m.license + ')' : '') +
          (m.note ? ': ' + m.note : '') + '.'
        : '';
      bigText.textContent = needBig
        ? ' Download ' + (m.name || m.id) + (sizeText(m.size_bytes) ? ' (' + sizeText(m.size_bytes) + ')' : '') + ' (over 500 MB).'
        : '';
    }

    function changed() {
      sync();
      say('', false);
      if (opts.onChange) opts.onChange(api_.getSelection());
    }

    function render() {
      select.textContent = '';
      var ph = el('option', null, models.length ? '(choose a model)' : '(no models available)');
      ph.value = '';
      select.appendChild(ph);
      var g = groupModels(models);
      GROUPS.forEach(function (gr) {
        if (!g[gr[0]].length) return;
        var og = el('optgroup');
        og.label = gr[1];
        g[gr[0]].forEach(function (m) {
          var o = el('option', null, optionText(m));
          o.value = m.id;
          og.appendChild(o);
        });
        select.appendChild(og);
      });
      renderDevices('');
      sync();
    }

    select.addEventListener('change', function () { renderDevices(device.value); changed(); });
    device.addEventListener('change', function () { if (opts.onChange) opts.onChange(api_.getSelection()); });
    lic.addEventListener('change', function () { say('', false); });
    big.addEventListener('change', function () { say('', false); });

    var api_ = {
      el: rootEl,
      select: select,
      say: say,
      defaultThreshold: function () { return data.default_threshold; },
      getSelection: function () {
        return { id: select.value, device: device.value, model: current() };
      },
      confirms: function () { return { license: lic.checked, large: big.checked }; },
      // '' when the selection can be submitted, else the message (also shown under the picker).
      problem: function () {
        var p = selectionProblem(current(), api_.confirms());
        say(p, !!p);
        return p;
      },
      setSelected: function (mid) { select.value = mid; if (select.value !== mid) select.value = ''; renderDevices(''); sync(); },
      ready: null
    };
    render();
    api_.ready = fetchModels().then(function (j) {
      models = j.models || [];
      data = { devices: j.devices || [], default_threshold: j.default_threshold };
      render();
      if (opts.onLoad) opts.onLoad(j, api_);
      return j;
    }).catch(function (e) {
      select.textContent = '';
      var o = el('option', null, '(models could not be read)');
      o.value = '';
      select.appendChild(o);
      say('The model list could not be read (' + e.message + ').', true);
      return null;
    });
    return api_;
  }

  root.ModelPicker = { create: create, fetchModels: fetchModels, util: util };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.ModelPicker;
})(typeof window !== 'undefined' ? window : globalThis);
