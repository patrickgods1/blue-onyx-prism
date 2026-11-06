// Train area, every tab: the status strip under the tabs (counts per tab and the one suggested next
// action, GET /v1/train/summary, filled after the page is up so a large dataset never holds it back),
// the Jobs tab's list of every job on the queue (GET /v1/train/jobs, every 3 s on that tab: step,
// percent, time left, a link to the job's row on its tab), the count on that tab's label, and the
// URL. A form's answer is the tab itself, rendered at the form's POST address: the address
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
  var nextLine = nextText.closest(".train-next");
  // This script runs above the tab's content: the Jobs card's list is looked up once it exists.
  var jobsBox = null, jobsEmpty = null;
  var jobsCount = document.getElementById("train-jobs-count");

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
    // The Jobs tab shows the job itself.
    nextLine.hidden = n.key === "job" && tab === "jobs";
    strip.hidden = false;
  }

  // The rows stay in place between polls, so a hover or keyboard focus survives.
  var rows = {};
  function span(cls, parent) {
    var el = document.createElement("span");
    el.className = cls;
    parent.appendChild(el);
    return el;
  }
  function jobRow() {
    var a = document.createElement("a");
    a.className = "tj";
    var head = span("tj-head", a);
    var f = { a: a, kind: span("tj-kind", head), title: span("tj-title", head), state: span("tj-state", a), open: span("tj-open", a) };
    var meter = span("tj-meter", a);
    f.bar = document.createElement("progress");
    f.bar.max = 100;
    meter.appendChild(f.bar);
    f.pct = span("tj-pct", meter);
    f.time = span("tj-time", a);
    f.step = span("tj-step", a);
    return f;
  }
  function fillRow(f, j) {
    var state = j.queued ? "Queued" : j.paused ? "Paused" : "Running";
    f.a.href = j.link;
    f.a.dataset.state = state.toLowerCase();
    f.a.setAttribute("aria-label", j.kind + " " + j.title + ", " + state + ", " + (j.queued ? "" : j.percent + "%, ") + j.time + ". Open the " + j.tab + " tab");
    f.kind.textContent = j.kind;
    f.title.textContent = j.title;
    f.title.title = j.title;
    f.state.textContent = state;
    f.time.textContent = j.time;
    f.bar.value = j.percent;
    f.pct.textContent = j.percent + "%";
    var where = j.step && j.phase ? j.step + ": " + j.phase : j.step || j.phase;
    var text = [where, j.detail].filter(Boolean).join(" · ");
    f.step.textContent = text;
    f.step.hidden = !text;
    f.open.textContent = j.tab + " ›";
  }
  function drawJobs(all) {
    var seen = {};
    all.forEach(function (j) {
      seen[j.id] = true;
      if (!rows[j.id]) rows[j.id] = jobRow();
      fillRow(rows[j.id], j);
    });
    Object.keys(rows).forEach(function (id) {
      if (!seen[id]) { rows[id].a.remove(); delete rows[id]; }
    });
    all.forEach(function (j) { jobsBox.appendChild(rows[j.id].a); });
    jobsBox.hidden = !all.length;
    jobsEmpty.hidden = !!all.length;
  }

  var jobsTimer = null, onQueue = null;
  function loadJobs() {
    if (jobsTimer) { clearTimeout(jobsTimer); jobsTimer = null; }
    fetch("/v1/train/jobs", { headers: { Accept: "application/json" }, cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j || !j.jobs) return;
        var all = j.jobs.active.concat(j.jobs.queued);
        var key = all.map(function (x) { return x.id; }).join();
        jobsCount.textContent = all.length;
        jobsCount.hidden = !all.length;
        if (jobsBox) drawJobs(all);
        // A job ended or started: the counts and the next step moved too.
        if (onQueue !== null && key !== onQueue) load();
        onQueue = key;
      })
      .catch(function () { /* the list stays as it was */ })
      .then(function () {
        if (jobsBox && !document.hidden) jobsTimer = setTimeout(loadJobs, 3000);
      });
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
  function loadAll() { load(); loadJobs(); }
  document.addEventListener("DOMContentLoaded", function () {
    jobsBox = document.getElementById("train-jobs");
    jobsEmpty = document.getElementById("train-jobs-empty");
    if (jobsBox) loadJobs();
  });
  loadAll();
  // Pages that change the counts say so.
  window.addEventListener("train-area:refresh", loadAll);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) loadAll(); });
  setInterval(function () { if (!document.hidden) loadAll(); }, 30000);
})();
