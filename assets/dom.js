// Shared DOM and fetch helpers of every page (QA-141): globalThis.PrismDom (el, clear, report)
// and globalThis.PrismFetch (json, postJson). Loaded by base.html after live.js, before any page
// script. Nothing here touches `document` or `fetch` at load, so node can load it; both are read
// when a helper is called.
//
// PrismFetch.json rejects (never resolves with an error answer) when the request fails, the
// answer is not ok, is not JSON or says `success: false`. The Error's message is the answer's
// `message` or `HTTP <status>`; it also carries `status` (the HTTP status), `body` (the parsed
// JSON, or null) and `conflict` (a 409, or `conflict` in the body). The pages' own wrappers that
// resolve (bench-core's PB.post / PB.getJSON) are not built on it.
'use strict';
(function () {
  // A new element; `cls` and `text` are skipped when null or undefined (`""` is text).
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  // Remove every child of `node`.
  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // Fill `box` with one line per entry of `lines` and, under them, a list of the first `max`
  // (default 50) `warnings` with "... n more" for the rest.
  function report(box, lines, warnings, max) {
    var cap = max == null ? 50 : max;
    box.textContent = '';
    lines.forEach(function (l) { box.appendChild(el('div', null, l)); });
    if (warnings && warnings.length) {
      var ul = el('ul', 'warn-list');
      warnings.slice(0, cap).forEach(function (w) { ul.appendChild(el('li', null, w)); });
      if (warnings.length > cap) ul.appendChild(el('li', null, '... ' + (warnings.length - cap) + ' more'));
      box.appendChild(ul);
    }
  }

  function json(url, init) {
    return fetch(url, init).then(function (r) {
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
    return json(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(body)
    });
  }

  globalThis.PrismDom = { el: el, clear: clear, report: report };
  globalThis.PrismFetch = { json: json, postJson: postJson };
})();
