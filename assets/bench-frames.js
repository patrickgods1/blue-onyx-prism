// Blue Onyx Prism: Benchmark frames with their boxes (Tune's strips and the Images grid; after
// bench-core.js). window.BenchFrames:
// - thumb(frame, opts): a button with the frame (GET /v1/benchmark/image, lazy) and its boxes in
//   an SVG overlay drawn in the image's own pixels (viewBox = width x height, the same fit as the
//   image): labels found (solid), missed (dashed), ignored (dotted); false alerts in their own
//   colour. A click opens the frame.
// - open(frame, opts): the dialog (templates/benchmark/image_dialog.html) with the full frame, the
//   boxes labelled, the facts, and "Open in Review" for a frame of Your cameras.
// `frame` is an example of GET /v1/benchmark/tune/images: {set, file, width, height, camera,
// scene, date, missed, falsePositives, gt [{label, bbox, ok, ignore}], preds [{label, confidence,
// bbox, ok}], review}. opts: {threshold, model, device}.
(function () {
  "use strict";
  if (typeof document === "undefined") return;
  var BC = window.BenchCore, PB = window.PrismBench, el = PB.el, fmt = BC.fmt;
  var SVGNS = "http://www.w3.org/2000/svg";
  function sv(tag, attrs, text) { var e = document.createElementNS(SVGNS, tag); Object.keys(attrs).forEach(function (k) { if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]); }); if (text !== undefined) e.textContent = text; return e; }
  function src(f) { return "/v1/benchmark/image?set=" + encodeURIComponent(f.set) + "&file=" + encodeURIComponent(f.file); }
  function boxClass(b, pred) {
    if (pred) return b.ok === false ? "fp" : null;
    if (b.ignore || b.ok === null || b.ok === undefined) return "ign";
    return b.ok ? "ok" : "miss";
  }
  // The overlay of `f`'s boxes; `labels` adds their text (the dialog; sized by fitLabels once the
  // dialog shows, as the text is in image pixels).
  function overlay(f, labels) {
    var w = f.width || 1, h = f.height || 1;
    var svg = sv("svg", {viewBox: "0 0 " + w + " " + h, preserveAspectRatio: "xMidYMid meet", "aria-hidden": "true", focusable: "false"});
    function draw(b, cls, text) {
      var r = b.bbox; if (!r) return;
      var x = Math.min(r[0], r[2]), y = Math.min(r[1], r[3]);
      svg.appendChild(sv("rect", {x: x, y: y, width: Math.abs(r[2] - r[0]), height: Math.abs(r[3] - r[1]), class: "bn-box " + cls}));
      if (labels && text) svg.appendChild(sv("text", {x: x, y: y, "data-box": [x, y, Math.max(r[0], r[2]), Math.max(r[1], r[3])].join(" "), class: "bn-label"}, text));
    }
    (f.gt || []).forEach(function (b) { var c = boxClass(b, false); draw(b, c, b.label + (c === "miss" ? " missed" : c === "ign" ? " (ignored)" : "")); });
    (f.preds || []).forEach(function (b) {
      var c = boxClass(b, true);
      if (c) draw(b, c, b.label + " " + fmt.n2(b.confidence));
      else if (labels && b.ok) draw(b, "ok", b.label + " " + fmt.n2(b.confidence));
    });
    if (labels) fitLabels(svg, w, h, 0, 0);
    return svg;
  }
  // The labels of `svg` (a frame of w x h) at BenchCore.labelSize for a box of boxW x boxH CSS
  // px (the font size and halo), placed by BenchCore.placeBoxLabels: inside the image and clear
  // of each other, each by its box (data-box: x1 y1 x2 y2), its width measured once sized.
  function fitLabels(svg, w, h, boxW, boxH) {
    var z = BC.labelSize(w, h, boxW, boxH), texts = Array.prototype.slice.call(svg.querySelectorAll("text.bn-label"));
    var items = texts.map(function (t) {
      t.setAttribute("font-size", z.size.toFixed(2));
      t.setAttribute("stroke-width", z.halo.toFixed(2));
      var b = String(t.getAttribute("data-box") || "0 0 0 0").split(" ").map(parseFloat), tw = null;
      try { tw = typeof t.getComputedTextLength === "function" ? t.getComputedTextLength() : null; } catch (e) { tw = null; }
      return {x1: b[0], y1: b[1], x2: b[2], y2: b[3], text: t.textContent, width: tw > 0 ? tw : null};
    });
    BC.placeBoxLabels(items, {w: w, h: h, size: z.size, gap: z.gap, halo: z.halo}).forEach(function (p, i) {
      texts[i].setAttribute("x", p.x.toFixed(2));
      texts[i].setAttribute("y", p.y.toFixed(2));
    });
  }
  function caption(f) { return [f.camera, f.scene, f.date].filter(Boolean).join(" · ") || f.file; }
  function describe(f) {
    var bits = [caption(f)];
    if (f.missed) bits.push(f.missed + " missed");
    if (f.falsePositives) bits.push(f.falsePositives + " false alert" + (f.falsePositives === 1 ? "" : "s"));
    return bits.join(", ");
  }
  function thumb(f, opts) {
    var b = el("button", {type: "button", class: "bn-thumb", "aria-label": "Open " + describe(f)});
    var img = el("img", {alt: "", loading: "lazy", decoding: "async", src: src(f)});
    img.addEventListener("error", function () { b.classList.add("broken"); }, {once: true});
    b.appendChild(img); b.appendChild(overlay(f, false));
    b.appendChild(el("span", {class: "bn-cap", text: caption(f)}));
    b.addEventListener("click", function () { open(f, opts || {}, b); });
    return b;
  }
  function fact(k, v) { return el("div", {}, [el("span", {class: "k", text: k}), el("span", {class: "v", text: v})]); }
  function counts(list) {
    var c = {};
    (list || []).forEach(function (b) { c[b.label] = (c[b.label] || 0) + 1; });
    return Object.keys(c).sort().map(function (k) { return c[k] + " " + k; }).join(", ") || "none";
  }
  var dlg = null, opener = null, shown = null;
  function refit() {
    var big = document.getElementById("img-big"), svg = big && big.querySelector("svg");
    if (shown && svg) fitLabels(svg, shown.width || 1, shown.height || 1, big.clientWidth, big.clientHeight);
  }
  function open(f, opts, from) {
    dlg = dlg || document.getElementById("img-dlg"); if (!dlg) return;
    opener = from || null;
    document.getElementById("img-title").textContent = caption(f);
    var big = document.getElementById("img-big");
    big.replaceChildren(el("img", {alt: describe(f), src: src(f)}), overlay(f, true));
    var facts = [fact("Camera", f.camera || "-"), fact("Scene", f.scene || "-"), fact("Date", f.date || "-"),
      fact("Labels", counts((f.gt || []).filter(function (b) { return !b.ignore; }))),
      fact("Detections" + (BC.has(opts.threshold) ? " at " + fmt.n2(opts.threshold) : ""), (f.preds || []).map(function (p) { return p.label + " " + fmt.n2(p.confidence); }).join(", ") || "none"),
      fact("Missed", String(f.missed || 0)), fact("False alerts", String(f.falsePositives || 0)),
      fact("Size", f.width && f.height ? f.width + " × " + f.height : "-"), fact("File", f.file)];
    if (opts.model) facts.unshift(fact("Model", opts.model + (opts.device ? " on " + opts.device : "")));
    document.getElementById("img-facts").replaceChildren.apply(document.getElementById("img-facts"), facts);
    var rv = document.getElementById("img-review");
    rv.hidden = !f.review;
    if (f.review) rv.href = "/train/review?frame=" + encodeURIComponent(f.review);
    if (!dlg.open) dlg.showModal();
    shown = f; refit();
  }
  function wire() {
    dlg = document.getElementById("img-dlg"); if (!dlg) return;
    document.getElementById("img-x").addEventListener("click", function () { dlg.close(); });
    dlg.addEventListener("close", function () {
      document.getElementById("img-big").replaceChildren(); shown = null;
      if (opener && opener.isConnected) opener.focus();
      opener = null;
    });
    // A click on the backdrop (outside the dialog's box) closes it.
    dlg.addEventListener("click", function (e) { if (e.target === dlg) dlg.close(); });
    // A turned phone or a resized window: the labels keep their size on screen.
    window.addEventListener("resize", function () { if (dlg.open) refit(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire); else wire();
  window.BenchFrames = {thumb: thumb, open: open, caption: caption};
})();
