// Blue Onyx Prism: Test page (served from /static/; no inline scripts under the CSP).
// Client-side flow: preview, drag-drop / paste, submit to the API with fetch and draw the boxes
// on a canvas. Without JavaScript the form posts to /test and the server draws the image.
(function () {
  var form = document.getElementById("test-form"), input = document.getElementById("image");
  var preview = document.getElementById("preview"), drop = document.getElementById("drop");
  var file = null, url = null;
  var COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080"];
  function setFile(f) {
    if (!f || !/^image\//.test(f.type)) return;
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
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    var lw = Math.max(2, Math.round(Math.min(c.width, c.height) / 300));
    ctx.lineWidth = lw;
    ctx.font = Math.max(12, lw * 7) + "px system-ui, sans-serif";
    ctx.textBaseline = "top";
    preds.forEach(function (p) {
      var col = colorFor(p.label);
      var w = p.x_max - p.x_min, h = p.y_max - p.y_min;
      ctx.strokeStyle = col;
      ctx.strokeRect(p.x_min, p.y_min, w, h);
      var text = p.label + " " + (p.confidence * 100).toFixed(0) + "%";
      var tw = ctx.measureText(text).width + 8, th = Math.max(12, lw * 7) + 6;
      var ty = p.y_min - th >= 0 ? p.y_min - th : p.y_min;
      ctx.fillStyle = col;
      ctx.fillRect(p.x_min, ty, tw, th);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, p.x_min + 4, ty + 3);
    });
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
    }
    document.getElementById("client-result").hidden = false;
    var img = new Image();
    img.onload = function () { draw(img, j.success ? (j.predictions || []) : []); };
    img.src = url;
  }
  form.addEventListener("submit", function (e) {
    if (!window.fetch || !window.FormData) return;
    if (!file && input.files[0]) setFile(input.files[0]);
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
    fetch(target, {method: "POST", body: fd}).then(function (r) { return r.json(); })
      .then(function (j) { show(j, model); })
      .catch(function (ex) {
        var err = document.getElementById("client-error");
        err.textContent = "Request failed: " + ex; err.hidden = false;
      })
      .finally(function () { busy.hidden = true; btn.disabled = false; });
  });
})();
