// Blue Onyx Prism: server-side file browser of the Training page's footage import
// (GET /v1/fs/browse?kind=video): folders and video files of the machine running the service.
// A button `button.browse[data-browse][data-target]` fills the input named by data-target of its
// form. Same dialog as the Config page's (served from /static/; no inline scripts under the CSP).
(function () {
  "use strict";
  var dlg = document.getElementById("browse-dialog");
  if (!window.fetch || !dlg || typeof dlg.showModal !== "function") return;
  var $ = function (id) { return document.getElementById(id); };
  var list = $("browse-list"), pathEl = $("browse-path"), msg = $("browse-msg"), roots = $("browse-roots");
  var up = $("browse-up"), use = $("browse-use"), title = $("browse-title");
  var kind = "video", target = null, cur = null, seq = 0;
  var TITLES = { video: "Choose a video or a folder", model: "Choose a model file or folder", classes: "Choose a classes YAML" };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  function pick(path) {
    target.value = path;
    target.dispatchEvent(new Event("input", { bubbles: true }));
    dlg.close();
    target.focus();
  }
  function render(j) {
    cur = j;
    pathEl.textContent = j.path;
    msg.textContent = j.truncated ? "Only the first entries are shown." : (j.entries.length ? "" : "No video here (choose this folder for its images).");
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
  document.querySelectorAll("button.browse").forEach(function (b) {
    b.hidden = false;
    b.addEventListener("click", function () {
      kind = TITLES[b.dataset.browse] ? b.dataset.browse : "video";
      target = b.form.elements[b.dataset.target];
      title.textContent = TITLES[kind];
      list.textContent = ""; msg.textContent = "Loading…";
      dlg.showModal();
      go(target.value.trim(), "");
    });
  });
  roots.addEventListener("change", function () { if (roots.value) go(roots.value, false); });
  up.addEventListener("click", function () { if (up.dataset.path) go(up.dataset.path, false); });
  use.addEventListener("click", function () { if (cur) pick(cur.path); });
  $("browse-cancel").addEventListener("click", function () { dlg.close(); });
})();
