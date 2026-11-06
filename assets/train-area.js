// Train area, every tab: the status strip under the tabs (counts per tab and the one suggested next
// action, GET /v1/train/summary, filled after the page is up so a large dataset never holds it back)
// and the URL. A form's answer is the tab itself, rendered at the form's POST address: the address
// bar goes to the tab's own URL, so a reload shows it again instead of asking to resubmit. An old
// /training#import link (the redirect keeps its fragment) goes on to the card's new tab, and a
// #fragment inside a closed section opens it.
(function () {
  "use strict";
  var strip = document.getElementById("train-strip");
  if (!strip) return;
  var tab = strip.dataset.tab;
  var home = "/train/" + tab;

  // Where the cards of the old Training page went.
  var MOVED = {
    import: "/train/import", autolabel: "/train/autolabel", review: "/train/review",
    use: "/train/models#use", compare: "/train/models#compare"
  };
  if (tab === "train" && MOVED[location.hash.slice(1)]) {
    location.replace(MOVED[location.hash.slice(1)]);
    return;
  }
  if (location.pathname !== home) {
    try { history.replaceState(null, "", home + location.hash); } catch (e) { /* stays */ }
  }

  function openTarget() {
    var id = location.hash.slice(1);
    var el = id && document.getElementById(id);
    if (!el) return;
    var opened = false;
    for (var d = el.closest("details"); d; d = d.parentElement && d.parentElement.closest("details")) {
      if (!d.open) { d.open = true; opened = true; }
    }
    if (opened) el.scrollIntoView();
  }
  document.addEventListener("DOMContentLoaded", openTarget);
  window.addEventListener("hashchange", openTarget);

  // On a phone the tabs scroll sideways: the current one in view.
  var nav = document.querySelector("nav.train-tabs"), active = nav && nav.querySelector("a.active");
  if (active) nav.scrollLeft = active.offsetLeft - nav.offsetLeft - (nav.clientWidth - active.offsetWidth) / 2;

  var list = document.getElementById("train-counts");
  var nextText = document.getElementById("train-next-text");
  var nextLink = document.getElementById("train-next-link");

  function item(tabId, number, label, note) {
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = "/train/" + tabId;
    var b = document.createElement("strong");
    b.textContent = number;
    a.appendChild(b);
    a.appendChild(document.createTextNode(" " + label));
    if (note) {
      var s = document.createElement("span");
      s.className = "muted";
      s.textContent = " " + note;
      a.appendChild(s);
    }
    li.appendChild(a);
    return li;
  }

  function draw(j) {
    var s = j.summary, n = j.next;
    var unknown = s.counted ? null : "…";
    list.replaceChildren(
      item("collect", s.frames_collected, "collected"),
      item("review", unknown || s.frames_to_review, "to review", s.boxes_to_review ? "(" + s.boxes_to_review + " boxes)" : ""),
      item("dataset", unknown || s.frames_reviewed, "reviewed"),
      item("dataset", s.exports, s.exports === 1 ? "export" : "exports"),
      item("train", s.runs, s.runs === 1 ? "run" : "runs"),
      item("models", s.models, s.models === 1 ? "model" : "models")
    );
    nextText.textContent = n.text;
    nextLink.textContent = n.label;
    nextLink.href = n.link;
    nextLink.hidden = n.tab === tab;
    strip.hidden = false;
  }

  var timer = null;
  function load() {
    if (timer) { clearTimeout(timer); timer = null; }
    fetch("/v1/train/summary", { headers: { Accept: "application/json" }, cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.summary) return;
        draw(j);
        // A first count of a large folder is still running.
        if (!j.summary.counted) timer = setTimeout(load, 2000);
      })
      .catch(function () { /* the strip stays as it was */ });
  }
  load();
  // Pages that change the counts say so.
  window.addEventListener("train-area:refresh", load);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) load(); });
  setInterval(function () { if (!document.hidden) load(); }, 30000);
})();
