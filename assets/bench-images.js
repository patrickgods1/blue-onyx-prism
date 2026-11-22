// Blue Onyx Prism: Benchmark Images (templates/benchmark/images.html; after bench-shell.js and
// bench-frames.js). One combo's scored frames at a confidence, from GET
// /v1/benchmark/tune/images (kind miss | fa | all, camera, scene, offset, limit): misses, false
// alerts, every frame, or the frames on which it disagrees with another combo (both combos' frames,
// compared here: BenchCore.disagree). The combos and sets come from GET /v1/benchmark/compare.
// A thumbnail opens the frame with its boxes, facts and, for Your cameras, "Open in Review".
// The address keeps every filter (history.replaceState).
(function () {
  "use strict";
  var BC = window.BenchCore, PB = window.PrismBench, S = window.BenchShell, F = window.BenchFrames, el = PB.el, fmt = BC.fmt;
  if (!S || !F) return;
  var $ = function (id) { return document.getElementById(id); };
  var PAGE = 48, DIS_PAGES = 10, DIS_LIMIT = 200;
  var q = new URLSearchParams(location.search);
  var st = {model: q.get("model") || "", device: q.get("device") || "", set: q.get("set") || "", threshold: parseFloat(q.get("threshold")),
    kind: q.get("kind") || "miss", camera: q.get("camera") || "all", scene: q.get("scene") || "all", other: q.get("other") || "",
    compare: null, offset: 0, frames: [], total: 0, dis: null};
  // One frames request (or "disagrees with" walk) and one Compare request at a time: the latest
  // choice wins (BenchCore.latest).
  var frameLoads = BC.latest(), compareLoads = BC.latest();
  if (["miss", "fa", "all", "dis"].indexOf(st.kind) < 0) st.kind = "miss";
  if (st.kind === "fp") st.kind = "fa";
  $("i-kind").value = st.kind; $("i-scene").value = ["all", "day", "night", "ir"].indexOf(st.scene) >= 0 ? st.scene : "all";
  function keyOf(m, d) { return m + "\u0000" + d; }
  function rowOf(k) { return ((st.compare && st.compare.rows) || []).filter(function (r) { return keyOf(r.model, r.device) === k; })[0]; }
  function err(text) { var e = $("i-error"); e.textContent = text || ""; e.hidden = !text; }
  function api(o) {
    var parts = [];
    Object.keys(o).forEach(function (k) { if (BC.has(o[k]) && o[k] !== "" && o[k] !== "all") parts.push(k + "=" + encodeURIComponent(o[k])); });
    return S.withGoal("/v1/benchmark/tune/images?" + parts.join("&"));
  }
  function remember() {
    var p = new URLSearchParams(location.search);
    [["model", st.model], ["device", st.device], ["set", st.set], ["threshold", BC.has(st.threshold) ? st.threshold.toFixed(2) : ""], ["kind", st.kind], ["camera", st.camera], ["scene", st.scene], ["other", st.kind === "dis" ? st.other : ""]].forEach(function (kv) {
      if (kv[1] && kv[1] !== "all") p.set(kv[0], kv[1]); else p.delete(kv[0]);
    });
    if (history.replaceState) history.replaceState(null, "", location.pathname + "?" + p.toString());
  }
  function option(v, t, on) { var o = el("option", {value: v, text: t}); o.selected = !!on; return o; }
  function renderSelects() {
    var rows = st.compare.rows || [];
    var combos = rows.map(function (r) { return option(keyOf(r.model, r.device), r.model + " · " + r.device, r.model === st.model && BC.sameDevice(r.device, st.device)); });
    $("i-combo").replaceChildren.apply($("i-combo"), combos.length ? combos : [option("", "No results yet")]);
    var others = rows.filter(function (r) { return !(r.model === st.model && BC.sameDevice(r.device, st.device)); });
    if (!others.some(function (r) { return keyOf(r.model, r.device) === st.other; })) st.other = others[0] ? keyOf(others[0].model, others[0].device) : "";
    $("i-other").replaceChildren.apply($("i-other"), others.map(function (r) { return option(keyOf(r.model, r.device), r.model + " · " + r.device, keyOf(r.model, r.device) === st.other); }));
    $("i-other-l").hidden = st.kind !== "dis";
    var sets = (st.compare.sets || []).map(function (s) { return option(s.id, s.title + " · " + fmt.int(s.frames) + " frames", s.id === st.set); });
    $("i-set").replaceChildren.apply($("i-set"), sets);
    $("i-thr").value = BC.has(st.threshold) ? st.threshold.toFixed(2) : "0.50";
  }
  function cameras(list) {
    var sel = $("i-cam"), keep = st.camera;
    sel.replaceChildren.apply(sel, [option("all", "All", keep === "all")].concat((list || []).map(function (c) { return option(c, c, c === keep); })));
    if (keep !== "all" && (list || []).indexOf(keep) < 0) { st.camera = "all"; sel.value = "all"; }
    sel.disabled = !(list && list.length);
  }
  function opts() { return {threshold: st.threshold, model: st.model, device: st.device}; }
  function show(append) {
    var grid = $("i-grid");
    var list = st.kind === "dis" ? st.dis.slice(append ? st.offset - PAGE : 0, st.offset) : st.frames.slice(append ? st.frames.length - st.lastPage : 0);
    var nodes = list.map(function (f) { return F.thumb(f, opts()); });
    if (append) nodes.forEach(function (n) { grid.appendChild(n); }); else grid.replaceChildren.apply(grid, nodes);
    var shown = grid.children.length, total = st.kind === "dis" ? st.dis.length : st.total;
    var what = {miss: "frames with misses", fa: "frames with false alerts", all: "frames", dis: "frames on which they disagree"}[st.kind];
    $("i-count").textContent = "· showing " + fmt.int(shown) + " of " + fmt.int(total) + " " + what + (st.kind === "all" || st.kind === "dis" ? "" : ", worst first") + " at " + fmt.n2(st.threshold);
    $("i-more-p").hidden = shown >= total;
    if (!total) grid.replaceChildren(el("p", {class: "bn-empty", text: st.kind === "miss" ? "Nothing missed at this confidence with these filters." : st.kind === "fa" ? "No false alert at this confidence with these filters." : "No frame matches these filters."}));
  }
  function base() { return {model: st.model, device: st.device, set: st.set, threshold: st.threshold.toFixed(2), camera: st.camera, scene: st.scene}; }
  function fetchPage(o, signal) { return PB.getJSON(api(o), signal ? {signal: signal} : {}); }
  function load(append) {
    if (!st.model || !BC.has(st.threshold)) { $("i-grid").removeAttribute("aria-busy"); return; }
    var t = frameLoads.begin(), signal = t.init.signal, grid = $("i-grid");
    err(""); remember();
    if (!append) { st.offset = 0; grid.setAttribute("aria-busy", "true"); if (!grid.children.length) grid.replaceChildren(el("span", {class: "bn-skel tall"})); }
    $("i-more").disabled = true;
    var done = function () { grid.removeAttribute("aria-busy"); $("i-more").disabled = false; };
    // A failure: the error, and no skeleton left in the grid.
    var failed = function (j) { done(); err(((j && j.message) || "The frames did not load.") + " Change a filter or reload the page."); if (grid.firstChild && /\bbn-skel\b/.test(grid.firstChild.className)) grid.replaceChildren(); };
    if (st.kind === "dis") {
      if (append) { t.wins(null); st.offset += PAGE; show(true); done(); return; }
      var other = rowOf(st.other);
      if (!other) { t.wins(null); done(); grid.replaceChildren(el("p", {class: "bn-empty", text: "Only one combo ran on these images: nothing to disagree with."})); $("i-count").textContent = ""; return; }
      var all = function (o) {
        var out = [], total = null;
        var page = function (k) {
          return fetchPage(Object.assign({}, o, {kind: "all", offset: k * DIS_LIMIT, limit: DIS_LIMIT}), signal).then(function (j) {
            if (!j.success) throw j;
            if (k === 0) cameras(j.images.cameras);
            out = out.concat(j.images.frames); total = j.images.total;
            return out.length < total && k + 1 < DIS_PAGES ? page(k + 1) : out;
          });
        };
        return page(0);
      };
      var o2 = Object.assign(base(), {model: other.model, device: other.device, threshold: (BC.has(other.confidence) ? other.confidence : st.threshold).toFixed(2)});
      Promise.all([all(base()), all(o2)]).then(function (r) {
        if (!t.wins(null)) return;
        st.dis = BC.disagree(r[0], r[1]); st.offset = PAGE; done(); err(""); show(false);
      }, function (j) { if (!t.wins(j)) return; failed(j); });
      return;
    }
    var o = Object.assign(base(), {kind: st.kind, offset: append ? st.frames.length : 0, limit: PAGE});
    fetchPage(o, signal).then(function (j) {
      if (!t.wins(j)) return; // a newer choice took its place
      if (!j.success) { failed(j); if (!append) grid.replaceChildren(); return; }
      done(); err("");
      cameras(j.images.cameras);
      st.total = j.images.total; st.lastPage = j.images.frames.length;
      st.frames = append ? st.frames.concat(j.images.frames) : j.images.frames;
      show(append);
    });
  }
  function start() {
    var t = compareLoads.begin();
    $("i-grid").setAttribute("aria-busy", "true"); // until the set's frames show
    PB.getJSON(S.withGoal("/v1/benchmark/compare" + (st.set ? "?set=" + encodeURIComponent(st.set) : "")), t.init).then(function (j) {
      if (!t.wins(j)) return; // a newer set took its place
      if (!j.success) {
        $("i-grid").removeAttribute("aria-busy");
        err(j._status === 404 ? "No results yet: run a benchmark first (step 1)." : (j.message || "The results did not load.") + " Reload the page.");
        return;
      }
      st.compare = j.compare; S.setGoal(j.compare.goal);
      if (!st.set && j.compare.set) st.set = j.compare.set.id;
      S.setSet(st.set);
      var rows = j.compare.rows || [];
      var r = rows.filter(function (x) { return x.model === st.model && (!st.device || BC.sameDevice(x.device, st.device)); })[0] || rows[0];
      if (r) { st.model = r.model; st.device = r.device; if (!BC.has(st.threshold)) st.threshold = BC.has(r.confidence) ? r.confidence : 0.5; }
      renderSelects();
      load(false);
    });
  }
  $("i-combo").addEventListener("change", function (e) { var r = rowOf(e.target.value); if (!r) return; st.model = r.model; st.device = r.device; if (BC.has(r.confidence)) st.threshold = r.confidence; renderSelects(); load(false); });
  // A set change: the frames in flight are of the set before.
  $("i-set").addEventListener("change", function (e) { st.set = e.target.value; st.camera = "all"; frameLoads.stop(); start(); });
  var thrTimer = null;
  $("i-thr").addEventListener("input", function (e) {
    var v = parseFloat(e.target.value);
    if (!(v >= 0.05 && v <= 1)) return;
    clearTimeout(thrTimer); thrTimer = setTimeout(function () { st.threshold = Math.round(v * 100) / 100; load(false); }, 400);
  });
  $("i-kind").addEventListener("change", function (e) { st.kind = e.target.value; $("i-other-l").hidden = st.kind !== "dis"; load(false); });
  $("i-other").addEventListener("change", function (e) { st.other = e.target.value; load(false); });
  $("i-cam").addEventListener("change", function (e) { st.camera = e.target.value; load(false); });
  $("i-scene").addEventListener("change", function (e) { st.scene = e.target.value; load(false); });
  $("i-more").addEventListener("click", function () { load(true); });
  S.onView(function (why) { if (why === "run" || why === "rescore") start(); });
  if (document.getElementById("bench").dataset.hasResults === "true") start();
  else err("No results yet: run a benchmark first (step 1, Run).");
})();
