// Blue Onyx Prism: Benchmark page, Compare tab (PrismBench.lineup): the models side by side on one
// set of images. The server (`lineup` of GET /v1/benchmark, benchmark::lineup) decides everything
// that is a judgement: which models are on a set, the shared objects, what is best past the noise,
// the plain sentence, who is not on these images and why. This file only draws it. The choices are
// remembered per browser; `?set=<id>` in the URL opens a set (links from the Train area).
// Loaded after bench-glossary.js, before benchmark.js, which calls render(status) on every poll
// (ES5, no build step, CSP script-src 'self').
(function () {
  var PB = window.PrismBench, L = window.PrismLive, el = PB.el, has = PB.has, F = PB.fmt;
  var $ = function (id) { return document.getElementById(id); };
  var saved = PB.store.get("lineup", null) || {};
  var wanted = null;
  try { wanted = new URLSearchParams(location.search).get("set"); } catch (e) { wanted = null; }
  var S = {j: null, set: saved.set || null, conf: saved.conf || "now", objects: saved.objects || "all", asked: wanted, chosen: !!saved.set};
  function remember() { PB.store.set("lineup", {set: S.chosen ? S.set : null, conf: S.conf, objects: S.objects}); }

  var SCENE_NAMES = {day: "Day", night: "Night", ir: "IR"};
  var OBJECT_NAMES = {all: "Everything scored", person: "People", vehicle: "Vehicles"};
  var CONF_NAMES = [["now", "Confidence each model uses now"], ["best", "Best confidence for your goal"]];
  var ROLE_TEXT = {held_out: "held-out", own_camera: "own camera", public: "public"};

  function currentSet(l) {
    var sets = l.sets || [];
    if (S.asked) {
      var a = sets.find(function (s) { return s.id === S.asked; });
      if (a) return {set: a, note: null};
    }
    var ask = S.asked, pick = sets.find(function (s) { return s.id === S.set; }) || sets.find(function (s) { return s.id === l.defaultSet; }) || sets[0] || null;
    return {set: pick, note: ask ? "That run's images have no comparison yet, so the default set is shown." : null};
  }
  function viewOf(s) { return (s.views || []).find(function (v) { return v.confidence === S.conf; }) || (s.views || [])[0]; }
  function objectKinds(v) {
    var seen = {all: true}, out = ["all"];
    (v ? v.rows : []).forEach(function (r) { Object.keys(r.objects || {}).forEach(function (k) { if (!seen[k]) { seen[k] = true; out.push(k); } }); });
    return out.sort(function (a, b) { return a === "all" ? -1 : b === "all" ? 1 : a < b ? -1 : 1; });
  }
  function select(label, name, opts, cur, onChange) {
    var l = el("label", "inline-label"), s = el("select"); s.name = name;
    l.appendChild(el("span", "lineup-label", label));
    opts.forEach(function (o) { var x = el("option", null, o[1]); x.value = o[0]; if (o[0] === cur) x.selected = true; s.appendChild(x); });
    s.addEventListener("change", function () { onChange(s.value); });
    l.appendChild(s); return l;
  }
  function setLabel(s) {
    var rows = (s.views && s.views[0] ? s.views[0].rows.length : 0);
    return s.title + " · " + s.images + " " + s.unit + " · " + rows + " model" + (rows === 1 ? "" : "s");
  }
  function tools(l, s, v) {
    var box = el("div", "lineup-tools");
    box.appendChild(select("Images", "lineup-set", l.sets.map(function (x) { return [x.id, setLabel(x)]; }), s.id, function (id) { S.set = id; S.asked = null; S.chosen = true; remember(); draw(); }));
    var kinds = objectKinds(v);
    if (kinds.length > 1) box.appendChild(select("Objects", "lineup-objects", kinds.map(function (k) { return [k, OBJECT_NAMES[k] || k]; }), kinds.indexOf(S.objects) >= 0 ? S.objects : "all", function (k) { S.objects = k; remember(); draw(); }));
    box.appendChild(select("Models at", "lineup-conf", CONF_NAMES, S.conf, function (c) { S.conf = c; remember(); draw(); }));
    return box;
  }
  // "23 frames, 18 objects, 5 empty. Your own frames, kept out of training. Scored on: ..."
  function description(s) {
    var p = el("p", "lineup-set-line");
    p.appendChild(el("strong", null, s.what + ": "));
    p.appendChild(document.createTextNode(s.images + " " + s.unit + ", " + s.objects + " object" + (s.objects === 1 ? "" : "s") + ", " + s.emptyFrames + " empty. Every model is scored on " + (s.classes.length ? s.classes.join(", ") : "no shared class") + (s.classes.indexOf("vehicle") >= 0 ? " (car, truck and bus as one)" : "") + "."));
    if (s.leftOut && s.leftOut.length) p.appendChild(el("span", "muted", " Left out, not every model has them: " + s.leftOut.join(", ") + "."));
    return p;
  }
  function pct1(c) { return has(c.value) ? (c.value * 100).toFixed(c.value === 1 || c.value === 0 ? 0 : 1) + "%" : "-"; }
  function mark(td, c) {
    if (c && c.best) { td.classList.add("lineup-best"); td.appendChild(el("span", "lineup-star", " best")); }
  }
  // One value cell; `label` is what a phone card shows beside it.
  function valueCell(tr, label, text, c, title, cls) {
    var td = el("td", "num" + (cls ? " " + cls : ""), text); td.dataset.label = label;
    if (!c || !has(c.value)) td.classList.add("lineup-none");
    if (title) td.title = title;
    mark(td, c); tr.appendChild(td); return td;
  }
  function scenesOf(s) { return [{id: "all", name: "All"}].concat((s.scenes || []).map(function (x) { return {id: x.id, name: SCENE_NAMES[x.id] || x.id}; })); }
  function headRow(s) {
    var scenes = scenesOf(s), one = scenes.length === 1, thead = el("thead"), a = el("tr"), b = el("tr");
    var m = el("th", "lineup-model", "Model"); m.rowSpan = 2; m.scope = "col"; a.appendChild(m);
    [["finds", "finds"], ["correct", "correct"]].forEach(function (g) {
      var th = el("th", "lineup-group"); th.appendChild(PB.head(g[1])); th.colSpan = scenes.length; th.scope = "colgroup"; if (one) th.rowSpan = 2; a.appendChild(th);
      if (!one) scenes.forEach(function (x) { b.appendChild(el("th", "num", x.name)); });
    });
    var fa = el("th", "lineup-group num"); fa.appendChild(PB.head("false_alerts")); fa.appendChild(el("span", "lineup-sub", "per 100 empty frames")); fa.rowSpan = 2; a.appendChild(fa);
    var sp = el("th", "lineup-group num"); sp.appendChild(PB.head("speed")); sp.appendChild(el("span", "lineup-sub", "ms per image"));
    if (!s.speed.comparable) { sp.classList.add("lineup-flag"); sp.appendChild(el("span", "tag restart", "not like for like")); sp.title = s.speed.reasons.join("; "); }
    sp.rowSpan = 2; a.appendChild(sp);
    thead.appendChild(a); if (!one) thead.appendChild(b);
    return thead;
  }
  function modelCell(r) {
    var td = el("td", "lineup-model");
    var name = el("a", "lineup-name", r.model); name.href = "#d-" + encodeURIComponent(r.model); name.title = "Every number of " + r.model + " in Details";
    td.appendChild(name);
    if (r.isDefault) { var t = el("span", "tag now", "Default"); t.title = "The model Blue Iris gets when it names none"; td.appendChild(t); }
    var c = PB.trainedChip(r.trainedOnShare); if (c) td.appendChild(c);
    var sub = el("span", "lineup-sub");
    sub.appendChild(document.createTextNode(r.device + " · at "));
    sub.appendChild(PB.conf(r.confidence));
    td.appendChild(sub);
    return td;
  }
  function row(s, r, kind) {
    var tr = el("tr");
    if (r.excluded) { tr.classList.add("lineup-out"); tr.title = "Not counted for best: it trained on some of these frames"; }
    tr.appendChild(modelCell(r));
    var o = (r.objects || {})[kind] || (r.objects || {}).all || {finds: {}, correct: {}};
    [["finds", "Finds"], ["correct", "Correct"]].forEach(function (g) {
      scenesOf(s).forEach(function (x) {
        var c = o[g[0]][x.id] || {};
        var what = g[0] === "finds" ? "of " + c.n + " objects" : "of " + c.n + " detections";
        valueCell(tr, g[1] + (x.id === "all" ? "" : " · " + x.name), pct1(c), c, has(c.value) ? Math.round(c.value * c.n) + " " + what : (g[0] === "finds" ? "No objects to find here" : "No detections here"));
      });
    });
    var fa = r.falseAlerts || {};
    var ft = valueCell(tr, "False alerts per 100", has(fa.value) ? (fa.value * 100).toFixed(fa.value === 0 ? 0 : 1) : "-", fa, has(fa.value) ? r.alerts + " of " + fa.n + " empty frames alerted" : "No empty frames in this set");
    if (has(fa.value)) ft.insertBefore(el("span", "lineup-count", r.alerts + " of " + fa.n + " "), ft.firstChild);
    var sp = r.speed || {};
    valueCell(tr, "Speed", has(sp.value) ? sp.value.toFixed(sp.value < 10 ? 1 : 0) + " ms" : "-", sp, sp.n ? "Median of " + sp.n + " images on " + r.device : "The run's overall median on " + r.device, s.speed.comparable ? null : "lineup-flag");
    return tr;
  }
  function table(s, v, kind) {
    var t = el("table", "bench stack lineup-table"); t.appendChild(headRow(s));
    var body = el("tbody"); v.rows.forEach(function (r) { body.appendChild(row(s, r, kind)); }); t.appendChild(body);
    var w = el("div", "scroll lineup-scroll"); w.dataset.keep = "w:lineup"; w.appendChild(t);
    return w;
  }
  function missing(s, j) {
    if (!s.missing || !s.missing.length) return null;
    var box = el("section", "lineup-missing"); box.appendChild(el("h3", null, "Not on these images"));
    var ul = el("ul", "lineup-missing-list");
    s.missing.forEach(function (m) {
      var li = el("li"); li.appendChild(el("strong", null, m.model)); li.appendChild(document.createTextNode(": " + m.why + ". "));
      if (m.blocked) li.appendChild(el("span", "muted", "Cannot run: " + m.blocked + "."));
      else {
        var b = el("button", "small secondary", "Run on these images"); b.type = "button"; b.disabled = !!j.running;
        if (j.running) b.title = "Available when the benchmark finishes";
        b.addEventListener("click", function () { if (PB.runOn) PB.runOn(m, s, b); });
        li.appendChild(b);
      }
      ul.appendChild(li);
    });
    box.appendChild(ul); return box;
  }
  function help(s) {
    var d = el("details", "help"); d.dataset.keep = "d:lineup-help"; d.appendChild(el("summary", null, "How to read this"));
    var n = PB.noise;
    [
      "Only models scored on exactly these images are in the table. Every model answers for the same objects, in the same classes, at the confidence shown under its name.",
      "Best marks a value only when the gap is real: accuracy past " + n.se_factor + " standard errors, speed past " + Math.round(n.speed_rel * 100) + "% and " + n.speed_abs_ms + " ms. Values inside that noise are level, so a column can have no best, or several.",
      "A dash means no data: nothing to find, no detection, or no empty frames in this set. A model with training frames in the set shows a chip and cannot be best, because it has seen those frames.",
      "These numbers are rescored for this table, so they can differ from Details, which scores each model on its own classes and images."
    ].forEach(function (t) { d.appendChild(el("p", null, t)); });
    if (s.relative) d.appendChild(el("p", null, "No hand labels on these images: scores are relative to " + (s.reference || "a reference model") + "."));
    return d;
  }
  function build() {
    var box = $("lineup"), j = S.j; if (!box) return;
    box.replaceChildren();
    var l = j && j.lineup, hasAny = l && l.sets && l.sets.length;
    if (!hasAny) {
      var models = ((j && j.results && j.results.models) || []).length;
      if (models) box.appendChild(el("p", "empty", "These results hold no scored images to compare (no labels, or the stored predictions are gone). Run a benchmark with labelled images under Run."));
      return;
    }
    var cur = currentSet(l), s = cur.set, v = viewOf(s);
    var card = el("section", "card primary lineup");
    card.appendChild(el("h2", null, "Which model does better on the same images?"));
    card.appendChild(tools(l, s, v));
    if (cur.note) card.appendChild(el("p", "muted", cur.note));
    card.appendChild(description(s));
    var call = v.headline.call, h = el("p", "lineup-headline lineup-" + call, v.headline.text); card.appendChild(h);
    if (s.relative) card.appendChild(el("p", "warn", "No hand labels on these images: scores are relative to " + (s.reference || "a reference model") + "."));
    if (!s.speed.comparable) card.appendChild(el("p", "warn", "Speed is not like for like here: " + s.speed.reasons.join("; ") + ". Accuracy still compares."));
    var kinds = objectKinds(v), kind = kinds.indexOf(S.objects) >= 0 ? S.objects : "all";
    card.appendChild(table(s, v, kind));
    var miss = missing(s, j); if (miss) card.appendChild(miss);
    card.appendChild(help(s));
    box.appendChild(card);
  }
  function draw() { var b = $("lineup"); if (b) L.keep(b, undefined, build); PB.refreshTip(); }
  function render(j) {
    S.j = j;
    var b = $("lineup"); if (!b) return;
    L.keep(b, JSON.stringify([j.lineup, S.set, S.asked, S.conf, S.objects, !!j.running, PB.noise]), build);
  }
  PB.lineup = {render: render};
})();
