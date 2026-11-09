// Blue Onyx Prism: Test page (served from /static/; no inline scripts under the CSP).
// Client-side flow: preview, drag-drop / paste, submit to the API with fetch and draw the boxes
// on a canvas. Without JavaScript the form posts to /test and the server draws the image.
(function () {
  var UPLOAD_TIMEOUT_MS = 60000;
  var form = document.getElementById("test-form"), input = document.getElementById("image");
  var preview = document.getElementById("preview"), drop = document.getElementById("drop");
  var file = null, url = null;
  var COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080"];
  var PREVIEW_MAX = 4096; // long side of the preview canvas; the upload keeps its size
  var pickError = document.getElementById("pick-error");
  if (window.matchMedia && matchMedia("(pointer: coarse)").matches) {
    document.getElementById("hint-fine").hidden = true;
    document.getElementById("hint-touch").hidden = false;
  }
  var IMAGE_EXT = /\.(webp|heic|heif|avif|gif|bmp|tiff?)$/i, picks = 0;
  function isImage(f) { return /^image\//.test(f.type) || (!f.type && IMAGE_EXT.test(f.name || "")); }
  function complain(text) { pickError.textContent = text; pickError.hidden = !text; }
  // The server decodes PNG and JPEG only: anything else (WebP, HEIC, GIF, AVIF...) is decoded by
  // the browser and re-encoded as JPEG before it is used.
  function toJpeg(f) {
    var stem = (f.name || "photo").replace(/\.[^.]*$/, "") || "photo";
    return createImageBitmap(f).then(function (bmp) {
      var c = document.createElement("canvas");
      c.width = bmp.width; c.height = bmp.height;
      c.getContext("2d").drawImage(bmp, 0, 0);
      if (bmp.close) bmp.close();
      return new Promise(function (resolve, reject) {
        c.toBlob(function (b) { if (b) resolve(new File([b], stem + ".jpg", {type: "image/jpeg"})); else reject(new Error("encode")); }, "image/jpeg", 0.92);
      });
    });
  }
  function setFile(f) {
    if (!f || !isImage(f)) return;
    var mine = ++picks;
    complain("");
    if (f.type === "image/png" || f.type === "image/jpeg") { useFile(f); return; }
    file = null; preview.hidden = true;
    if (!window.createImageBitmap) { complain("This browser cannot convert " + (f.type || "that file") + ". Choose a PNG or JPEG photo (the server reads those two formats)."); return; }
    toJpeg(f).then(function (jpg) { if (mine === picks) useFile(jpg); }, function () {
      if (mine === picks) complain("This browser could not read " + (f.name || "that file") + " as a picture. The server reads PNG and JPEG only: choose one of those.");
    });
  }
  function useFile(f) {
    file = f;
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(f);
    preview.src = url; preview.hidden = false;
    // Keep the file input in sync so the no-fetch fallback still works.
    try { var dt = new DataTransfer(); dt.items.add(f); input.files = dt.files; } catch (e) {}
  }
  input.addEventListener("change", function () { setFile(input.files[0]); });
  ["dragenter", "dragover"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); });
  });
  drop.addEventListener("drop", function (e) {
    if (e.dataTransfer && e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", function (e) {
    var items = (e.clipboardData && e.clipboardData.items) || [];
    for (var i = 0; i < items.length; i++) {
      if (items[i].kind === "file" && /^image\//.test(items[i].type)) { setFile(items[i].getAsFile()); e.preventDefault(); return; }
    }
  });
  function colorFor(label) {
    var h = 0;
    for (var i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
    return COLORS[h % COLORS.length];
  }
  function draw(img, preds) {
    var c = document.getElementById("canvas"), ctx = c.getContext("2d");
    // The preview is at most PREVIEW_MAX px on its long side; the server's boxes are in the
    // photo's own pixels, so they are scaled by the same factor.
    var k = Math.min(1, PREVIEW_MAX / Math.max(img.naturalWidth, img.naturalHeight));
    c.width = Math.max(1, Math.round(img.naturalWidth * k)); c.height = Math.max(1, Math.round(img.naturalHeight * k));
    var kx = c.width / img.naturalWidth, ky = c.height / img.naturalHeight;
    // style.css sizes the canvas from these, keeping the image's ratio.
    c.style.setProperty("--ar", String(c.width / c.height));
    c.style.setProperty("--nw", c.width + "px");
    ctx.drawImage(img, 0, 0, c.width, c.height);
    var lw = Math.max(2, Math.round(Math.min(c.width, c.height) / 300));
    ctx.lineWidth = lw;
    ctx.font = Math.max(12, lw * 7) + "px system-ui, sans-serif";
    ctx.textBaseline = "top";
    preds.forEach(function (p) {
      var col = colorFor(p.label);
      var x0 = p.x_min * kx, y0 = p.y_min * ky;
      ctx.strokeStyle = col;
      ctx.strokeRect(x0, y0, (p.x_max - p.x_min) * kx, (p.y_max - p.y_min) * ky);
      var text = p.label + " " + (p.confidence * 100).toFixed(0) + "%";
      var tw = ctx.measureText(text).width + 8, th = Math.max(12, lw * 7) + 6;
      var ty = y0 - th >= 0 ? y0 - th : y0;
      ctx.fillStyle = col;
      ctx.fillRect(x0, ty, tw, th);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x0 + 4, ty + 3);
    });
  }
  // `effective_threshold` of src/worker/mod.rs: a request value over 1 is a percentage, over 100
  // it is ignored; never under the 0.01 floor.
  var FLOOR = 0.01;
  function effective(asked, configured) {
    var t = asked > 0 && isFinite(asked) ? (asked <= 1 ? asked : asked <= 100 ? asked / 100 : configured) : configured;
    return isFinite(t) && t >= FLOOR ? t : FLOOR;
  }
  function fmtT(t) { return t.toFixed(3).replace(/0$/, ""); }
  // Twin of `no_objects_line` (src/server/pages.rs): what an answer with no predictions says,
  // [sentence, hint]. The model's thresholds come from the select's data-thresholds.
  function noneLine(model) {
    var field = form.querySelector('[name="min_confidence"]'), asked = parseFloat(field ? field.value : "");
    var map = {}, sel = document.getElementById("model");
    try { map = JSON.parse((sel && sel.dataset.thresholds) || "{}") || {}; } catch (e) { map = {}; }
    var m = map[model], t, night = null;
    if (asked > 0 && asked <= 100) t = effective(asked, 0);
    else if (m && typeof m.t === "number") {
      t = effective(0, m.t);
      if (typeof m.night === "number") night = effective(0, m.night);
    } else return ["No objects found.", "Lower Min confidence to see weaker detections."];
    if (night !== null && fmtT(night) === fmtT(t)) night = null;
    var line = "No objects at or above " + fmtT(t) + (night !== null ? " (" + fmtT(night) + " on night and IR frames)" : "") + ".";
    return [line, Math.min(t, night === null ? t : night) > FLOOR ? "Lower Min confidence to see weaker detections." : ""];
  }
  function show(j, model) {
    var err = document.getElementById("client-error");
    document.getElementById("client-json").textContent = JSON.stringify(j, null, 2);
    var sum = document.getElementById("client-summary"), tbody = document.querySelector("#detections tbody");
    tbody.replaceChildren();
    if (!j.success) {
      err.textContent = "Detection failed: " + (j.message || j.error || "unknown error");
      err.hidden = false;
      sum.textContent = "";
    } else {
      err.hidden = true;
      sum.textContent = j.count + " object(s) from model '" + model + "' on " + j.executionProvider +
        ": inference " + j.inferenceMs + " ms, process " + j.processMs + " ms, round trip " + j.analysisRoundTripMs + " ms.";
      (j.predictions || []).forEach(function (p) {
        var tr = document.createElement("tr");
        [p.label, (p.confidence * 100).toFixed(1) + "%", p.x_min + ", " + p.y_min + ", " + p.x_max + ", " + p.y_max].forEach(function (t, i) {
          var td = document.createElement("td"); td.textContent = t;
          if (i > 0) td.className = "num";
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      if (!(j.predictions || []).length) {
        var none = noneLine(model), tr = document.createElement("tr"), td = document.createElement("td");
        td.colSpan = 3; td.className = "zero-state"; td.textContent = none[0];
        if (none[1]) {
          var hint = document.createElement("span"); hint.className = "muted"; hint.textContent = " " + none[1];
          td.appendChild(hint);
        }
        tr.appendChild(td); tbody.appendChild(tr);
      }
    }
    document.getElementById("client-result").hidden = false;
    var img = new Image();
    img.onload = function () { draw(img, j.success ? (j.predictions || []) : []); };
    img.src = url;
  }
  form.addEventListener("submit", function (e) {
    if (!window.fetch || !window.FormData) return;
    if (!file && input.files[0]) { e.preventDefault(); if (pickError.hidden) setFile(input.files[0]); return; }
    if (!file) return;
    e.preventDefault();
    var model = document.getElementById("model").value;
    var fd = new FormData(form);
    fd.set("image", file, file.name || "pasted.png");
    var target = model ? "/v1/vision/custom/" + encodeURIComponent(model) : "/v1/vision/detection";
    var busy = document.getElementById("busy"), btn = document.getElementById("detect");
    busy.hidden = false; btn.disabled = true;
    var old = document.getElementById("server-result");
    if (old) old.remove();
    // An upload that gets no answer in UPLOAD_TIMEOUT_MS is stopped (a network switch can leave
    // it hanging for good). The clock waits while the sign-in dialog is open (live.js repeats the
    // upload once signed in).
    var ctl = window.AbortController ? new AbortController() : null, timer = null, timedOut = false;
    function arm() {
      if (!ctl) return;
      timer = setTimeout(function () {
        if (window.PrismPoll && PrismPoll.signingIn && PrismPoll.signingIn()) { arm(); return; }
        timedOut = true; ctl.abort();
      }, UPLOAD_TIMEOUT_MS);
    }
    arm();
    // The header keeps a test upload out of the dataset: it is no camera's frame.
    fetch(target, {method: "POST", body: fd, headers: {"x-prism-no-collect": "1"}, signal: ctl ? ctl.signal : undefined}).then(function (r) { return r.json(); })
      .then(function (j) { show(j, model); })
      .catch(function (ex) {
        var err = document.getElementById("client-error");
        err.textContent = timedOut
          ? "No answer within " + (UPLOAD_TIMEOUT_MS / 1000) + " s, so the upload was stopped. Check that the server runs (and this device's network), then press Detect again."
          : "Request failed: " + ex;
        err.hidden = false;
      })
      .finally(function () { clearTimeout(timer); busy.hidden = true; btn.disabled = false; });
  });
})();
