// Blue Onyx Prism: the Browse... buttons of the Training page's footage import (data-browse=video
// or folder) and the Dataset page's folder field (folder). A button
// `button.browse[data-browse][data-target]` fills the input named by data-target of its form.
// Native first: when GET /v1/fs/pick says a desktop is reachable (this browser is on the server's
// machine and the service can show a dialog), a click POSTs /v1/fs/pick and the OS file/folder
// dialog opens on the server's screen. Without one (or when that answers 409 / fails), the
// in-page server-side browser opens instead (GET /v1/fs/browse, the <dialog id="browse-dialog">
// of the page; same dialog as the Config page's). Served from /static/ (no inline scripts under
// the CSP).
(function () {
  "use strict";
  if (!window.fetch) return;
  var $ = function (id) { return document.getElementById(id); };
  var dlg = $("browse-dialog");
  var inPage = !!(dlg && typeof dlg.showModal === "function");
  var buttons = document.querySelectorAll("button.browse[data-browse][data-target]");
  if (!buttons.length) return;
  var KINDS = { video: 1, folder: 1, model: 1, classes: 1 };
  // null until GET /v1/fs/pick answers; a click before that uses the in-page browser.
  var native = null;
  fetch("/v1/fs/pick", { headers: { Accept: "application/json" } }).then(function (r) {
    return r.ok ? r.json() : null;
  }).then(function (j) { native = !!(j && j.available); if (native) reveal(); }).catch(function () { native = false; });

  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  function fill(target, path) {
    target.value = path;
    target.dispatchEvent(new Event("input", { bubbles: true }));
    target.focus();
  }

  // The in-page browser (server-side listing).
  var list, pathEl, msg, roots, up, use, title;
  var kind = "video", target = null, cur = null, seq = 0;
  var TITLES = { video: "Choose a video or a folder", model: "Choose a model file or folder", classes: "Choose a classes YAML", folder: "Choose the dataset folder" };
  var EMPTY = { video: "No video here (choose this folder for its images).", folder: "No subfolders here.", model: "No model files here.", classes: "No classes YAML here." };
  function pick(path) {
    dlg.close();
    fill(target, path);
  }
  function render(j) {
    cur = j;
    pathEl.textContent = j.path;
    msg.textContent = j.truncated ? "Only the first entries are shown." : (j.entries.length ? "" : EMPTY[kind]);
    up.disabled = !j.parent;
    up.dataset.path = j.parent || "";
    roots.textContent = "";
    var here = el("option", null, "Places…"); here.value = ""; roots.appendChild(here);
    j.roots.forEach(function (r) { var o = el("option", null, r.label + "  " + r.path); o.value = r.path; roots.appendChild(o); });
    roots.value = "";
    use.hidden = kind === "classes";
    list.textContent = "";
    var sel = null;
    j.entries.forEach(function (e) {
      var b = el("button", "browse-item" + (e.dir ? " dir" : " file"));
      b.type = "button";
      b.appendChild(el("span", "browse-name", (e.dir ? "📁 " : "") + e.name));
      b.appendChild(el("span", "browse-size", e.dir ? "" : e.sizeText));
      if (!e.dir && j.selected === e.name) { b.classList.add("selected"); sel = b; }
      b.addEventListener("click", function () { if (e.dir) go(e.path, false); else pick(e.path); });
      list.appendChild(b);
    });
    if (sel && sel.scrollIntoView) sel.scrollIntoView({ block: "nearest" });
  }
  function go(path, retry) {
    var n = ++seq;
    var url = "/v1/fs/browse?kind=" + encodeURIComponent(kind) + (path ? "&path=" + encodeURIComponent(path) : "");
    fetch(url, { headers: { Accept: "application/json" } }).then(function (r) {
      return r.json().then(function (j) { return { status: r.status, body: j }; });
    }).then(function (res) {
      if (n !== seq) return;
      if (res.status === 200) { render(res.body); if (retry) msg.textContent = retry; return; }
      var m = (res.body && res.body.message) || "Cannot list that folder.";
      if (res.status === 404 && path) { go("", m); return; }
      msg.textContent = m;
    }).catch(function () { if (n === seq) msg.textContent = "Cannot reach the server."; });
  }
  function openInPage(b, t) {
    if (!inPage) return;
    kind = TITLES[b.dataset.browse] ? b.dataset.browse : "video";
    target = t;
    title.textContent = TITLES[kind];
    list.textContent = ""; msg.textContent = "Loading…";
    dlg.showModal();
    go(target.value.trim(), "");
  }
  if (inPage) {
    list = $("browse-list"); pathEl = $("browse-path"); msg = $("browse-msg"); roots = $("browse-roots");
    up = $("browse-up"); use = $("browse-use"); title = $("browse-title");
    roots.addEventListener("change", function () { if (roots.value) go(roots.value, false); });
    up.addEventListener("click", function () { if (up.dataset.path) go(up.dataset.path, false); });
    use.addEventListener("click", function () { if (cur) pick(cur.path); });
    $("browse-cancel").addEventListener("click", function () { dlg.close(); });
  }

  // The native dialog on the server's screen.
  function statusFor(b) {
    if (b._browseStatus) return b._browseStatus;
    var s = el("span", "browse-status muted");
    s.setAttribute("role", "status");
    var after = (b.closest && b.closest(".with-browse")) || b;
    after.parentNode.insertBefore(s, after.nextSibling);
    b._browseStatus = s;
    return s;
  }
  function openNative(b, t) {
    var k = KINDS[b.dataset.browse] ? b.dataset.browse : "video";
    var status = statusFor(b);
    b.disabled = true;
    status.textContent = " Waiting for the file dialog on the server's screen…";
    var done = function (text) { b.disabled = false; status.textContent = text || ""; };
    fetch("/v1/fs/pick", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ kind: k, start: t.value.trim() })
    }).then(function (r) {
      return r.json().then(function (j) { return { status: r.status, body: j || {} }; }, function () { return { status: r.status, body: {} }; });
    }).then(function (res) {
      var j = res.body;
      if (res.status === 200 && typeof j.path === "string" && j.path) { done(""); fill(t, j.path); return; }
      if (res.status === 200 && j.cancelled) { done(""); return; }
      // 409 (no desktop, a remote browser, the dialog failed, busy) or anything else: the
      // in-page browser. Only "busy" may clear up, so only it keeps the native path.
      if (res.status === 409 && j.available === false) native = false;
      done(j.busy && j.message ? " " + j.message : "");
      openInPage(b, t);
    }).catch(function () { done(""); openInPage(b, t); });
  }

  // Without the in-page dialog a button shows only once the native one is known to work.
  function reveal() {
    Array.prototype.forEach.call(buttons, function (b) { if (b._browseTarget) b.hidden = false; });
  }
  Array.prototype.forEach.call(buttons, function (b) {
    var t = b.form && b.form.elements[b.dataset.target];
    if (!t) return;
    b._browseTarget = t;
    b.addEventListener("click", function () {
      if (native) openNative(b, t); else openInPage(b, t);
    });
  });
  if (inPage) reveal();
})();
