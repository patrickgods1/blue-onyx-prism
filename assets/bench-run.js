// Blue Onyx Prism: Benchmark step 1, Run (templates/benchmark/run.html; after bench-shell.js).
// The form (images, models, devices, options) posts POST /v1/benchmark (Start) or
// POST /v1/benchmark/settings (Save as default). Rules:
// - Two GPU devices ticked: the warning under the devices, and Start asks to confirm in the page
//   (a sweep of every GPU device has hung a machine; the server refuses it without
//   `confirm_gpus`).
// - Two or more devices ticked and no CPU device: the CPU reference is ticked for the user with a
//   note and Undo (BenchCore.cpuReference): without it no device is recommended.
// - The estimate: the last run's time per image of each ticked model on each ticked device
//   (BenchCore.estimate), with the burst test's time per combo when it is ticked
//   (BenchCore.burstSeconds; its three numbers are off while it is unticked).
// - While a run goes: its combos and their state from the run poll (BenchCore.comboStates); Start
//   is off; at the end "Compare results".
// Downloads (POST /v1/resources/download) follow GET /v1/resources until done, then reload.
(function () {
  "use strict";
  var BC = window.BenchCore, PB = window.PrismBench, S = window.BenchShell, el = PB.el, fmt = BC.fmt;
  if (!S) return;
  var $ = function (id) { return document.getElementById(id); };
  var form = $("run-form"), start = $("start"), saveDef = $("save-def"), msg = $("run-msg");
  form.addEventListener("submit", function (e) { e.preventDefault(); });
  function say(text, kind) { msg.className = "bn-msg" + (kind === "error" ? " bn-error" : kind === "ok" ? " notice" : ""); msg.textContent = text || ""; }
  function boxes(name) { return Array.prototype.slice.call(form.querySelectorAll("input[name=" + name + "]")); }
  function ticked(name) { return boxes(name).filter(function (c) { return c.checked && !c.disabled; }); }

  // ---- GPU confirmation and the CPU reference ------------------------------------------------------
  var gpuWarn = $("gpu-warn"), cpuNote = $("cpu-note"), autoCpu = null, dismissed = null;
  function devices() { return boxes("device").map(function (c) { return {spec: c.value, checked: c.checked, runnable: !c.disabled, cpu: c.dataset.cpu === "1", box: c}; }); }
  function gpusTicked() { return ticked("device").filter(function (c) { return c.dataset.gpu === "1"; }); }
  function deviceRules(changed) {
    if (changed && changed === autoCpu && !changed.checked) { dismissed = changed.value; autoCpu = null; cpuNote.hidden = true; }
    if (autoCpu && autoCpu.checked) {
      // Back to one device, or another CPU device ticked: the added reference is not needed.
      var others = ticked("device").filter(function (c) { return c !== autoCpu; });
      if (others.length < 2 || others.some(function (c) { return c.dataset.cpu === "1"; })) { autoCpu.checked = false; autoCpu = null; cpuNote.hidden = true; }
    }
    // Once the user took the added reference away, none is added again (of either runtime).
    var pick = dismissed ? null : BC.cpuReference(devices(), null);
    var box = pick && boxes("device").filter(function (c) { return c.value === pick; })[0];
    if (box) {
      box.checked = true; autoCpu = box;
      $("cpu-note-text").textContent = "Ticked " + pick + " as the CPU reference: without it no device can be checked against the CPU or recommended.";
      cpuNote.hidden = false;
    }
    gpuWarn.hidden = gpusTicked().length < 2;
    if (gpusTicked().length < 2) $("gpu-confirm").hidden = true;
  }
  $("cpu-undo").addEventListener("click", function () {
    if (autoCpu) { autoCpu.checked = false; dismissed = autoCpu.value; autoCpu = null; }
    cpuNote.hidden = true; deviceRules(); estimate();
  });

  // ---- estimate ------------------------------------------------------------------------------------
  function images() {
    var max = Math.max(0, parseInt($("o-max").value, 10) || 0), n = 0;
    ticked("dataset").forEach(function (c) { var k = parseInt(c.dataset.images, 10) || 0; n += max > 0 && k > 0 ? Math.min(max, k) : k; });
    if ($("add-folder").value.trim()) n += max || 0;
    return n;
  }
  function burstLoad() {
    if (!$("o-burst").checked) return null;
    return {cameras: parseInt($("o-burst-cameras").value, 10) || 1, images: parseInt($("o-burst-images").value, 10) || 1, interval_ms: parseInt($("o-burst-interval").value, 10) || 0};
  }
  function burstNote(burst, models, devs, timing) {
    var note = $("o-burst-est");
    if (!burst) { note.textContent = "Off: each image is timed one at a time."; return; }
    var lo = null, hi = null;
    models.forEach(function (m) { devs.forEach(function (d) {
      var t = timing[m.value] || {}, per = null;
      Object.keys(t).forEach(function (k) { if (BC.sameDevice(k, d.value)) per = t[k]; });
      var sec = BC.burstSeconds(burst, per);
      lo = lo === null ? sec : Math.min(lo, sec); hi = hi === null ? sec : Math.max(hi, sec);
    }); });
    if (lo === null) lo = hi = BC.burstSeconds(burst, null);
    var each = burst.cameras * burst.images * 3;
    note.textContent = each + " requests per combo; about " + (Math.round(lo / 10) === Math.round(hi / 10) ? fmt.duration(hi) : fmt.duration(lo) + " to " + fmt.duration(hi)) + " per combo, added to the estimate below.";
  }
  function estimate() {
    var models = ticked("model"), devs = ticked("device"), timing = {};
    models.forEach(function (c) { try { timing[c.value] = JSON.parse(c.dataset.timing || "{}"); } catch (e) { timing[c.value] = {}; } });
    var n = images(), burst = burstLoad();
    $("o-burst-fields").disabled = !burst;
    burstNote(burst, models, devs, timing);
    var e = BC.estimate({models: models.map(function (c) { return c.value; }), devices: devs.map(function (c) { return c.value; }), timing: timing, images: n,
      repeat: parseInt($("o-repeat").value, 10) || 1, warmup: parseInt($("o-warmup").value, 10) || 0, burst: burst});
    var what = models.length + " model" + (models.length === 1 ? "" : "s") + " × " + devs.length + " device" + (devs.length === 1 ? "" : "s") + " × " + fmt.int(n) + " images";
    if (!e.pairs || !n) { $("est").textContent = "Tick images, models and devices"; $("est-detail").textContent = ""; return; }
    if (!BC.has(e.seconds)) { $("est").textContent = "Time unknown"; $("est-detail").textContent = "for " + what + " (no earlier run of these models on these devices)"; return; }
    $("est").textContent = "About " + fmt.duration(e.seconds);
    $("est-detail").textContent = "for " + what + (BC.has(e.burst) && e.burst > 0 ? ", burst test " + fmt.duration(e.burst) + " of it" : "") + (e.known < e.pairs ? " (" + e.known + " of " + e.pairs + " pairs ran before; the others add to it)" : "") + ". Downloads and the reference model add to it.";
  }
  form.addEventListener("change", function (e) { if (e.target && e.target.name === "device") deviceRules(e.target); estimate(); });
  form.addEventListener("input", estimate);
  ["o-max", "o-repeat", "o-warmup", "o-burst-cameras", "o-burst-images", "o-burst-interval"].forEach(function (id) { $(id).addEventListener("input", estimate); });
  $("o-burst").addEventListener("change", estimate);

  // ---- the form's body ---------------------------------------------------------------------------
  function folderField() { var f = $("add-folder").value.trim(); return f ? (f.indexOf("dir:") === 0 ? f : "dir:" + f) : null; }
  function body(forSettings) {
    var b = new URLSearchParams();
    ticked("model").forEach(function (c) { b.append("model", c.value); });
    ticked("device").forEach(function (c) { b.append("device", c.value); });
    ticked("dataset").forEach(function (c) { b.append("dataset", c.value); });
    var f = folderField();
    if (f) b.append(forSettings ? "add_folder" : "dataset", forSettings ? $("add-folder").value.trim() : f);
    b.append("max_images", $("o-max").value || "0");
    b.append("repeat", $("o-repeat").value || "1");
    b.append("warmup", $("o-warmup").value || "0");
    b.append("low_priority", $("o-low").checked ? "1" : "0");
    b.append("served", $("o-served").checked ? "1" : "0");
    b.append("burst", $("o-burst").checked ? "1" : "0");
    b.append("burst_cameras", $("o-burst-cameras").value || "");
    b.append("burst_images", $("o-burst-images").value || "");
    b.append("burst_interval_ms", $("o-burst-interval").value || "");
    b.append("reference_model", $("o-ref").value);
    return b;
  }
  function problem() {
    if (!ticked("model").length) return "Tick at least one model.";
    if (!ticked("device").length) return "Tick at least one device to benchmark on.";
    if (!ticked("dataset").length && !folderField()) return "Tick at least one image set (or add a folder).";
    return null;
  }

  // ---- Start ---------------------------------------------------------------------------------------
  var confirmBox = $("gpu-confirm");
  function go(confirmGpus) {
    var b = body(false);
    if (confirmGpus) b.append("confirm_gpus", "1");
    start.disabled = true; saveDef.disabled = true; say("Starting…");
    PB.post("/v1/benchmark", b).then(function (j) {
      if (j.success) { say("Started: the progress shows above and this run's combos below. Compare shows the results when it ends.", "ok"); S.kick(); }
      else { say((j.message || "The benchmark did not start.") + (j._status === 409 ? "" : " Check the form and try again."), "error"); start.disabled = false; }
      saveDef.disabled = false;
    });
  }
  start.addEventListener("click", function () {
    var p = problem(); if (p) { say(p, "error"); return; }
    var gpus = gpusTicked();
    if (gpus.length > 1) {
      $("gpu-confirm-text").textContent = gpus.length + " GPU devices are ticked (" + gpus.map(function (c) { return c.value; }).join(", ") + "). " + gpuWarn.textContent + " Run them one after the other anyway?";
      confirmBox.hidden = false; $("gpu-yes").focus(); return;
    }
    go(false);
  });
  $("gpu-yes").addEventListener("click", function () { confirmBox.hidden = true; go(true); });
  $("gpu-no").addEventListener("click", function () { confirmBox.hidden = true; start.focus(); });

  // ---- Save as default -----------------------------------------------------------------------------
  saveDef.addEventListener("click", function () {
    var p = problem(); if (p) { say(p, "error"); return; }
    var hadFolder = !!folderField();
    saveDef.disabled = true; say("Saving the defaults…");
    PB.post("/v1/benchmark/settings", body(true)).then(function (j) {
      say(j.message || (j.success ? "Saved as the default." : "Not saved."), j.success ? "ok" : "error");
      if (j.success && hadFolder) window.PrismLive.reloadKeep();
      saveDef.disabled = false;
    });
  });

  // ---- downloads and folders -----------------------------------------------------------------------
  form.querySelectorAll("[data-download]").forEach(function (b) {
    b.addEventListener("click", function () {
      var id = b.dataset.download; b.disabled = true; b.textContent = "Starting…";
      PB.post("/v1/resources/download", new URLSearchParams({id: id})).then(function (j) {
        if (!j.success) { say(j.message || "The download did not start.", "error"); b.disabled = false; b.textContent = "Download"; return; }
        say(j.message || "Downloading.", "ok");
        var poll = PrismPoll.create({period: 1500, max: 30000, run: function () {
          return PrismPoll.fetch("/v1/resources", {cache: "no-store"}).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }).then(function (k) {
            var row = (k.resources || []).filter(function (x) { return x.id === id; })[0];
            if (row && row.busy) { b.textContent = row.state_text || "Downloading…"; return true; }
            window.PrismLive.reloadKeep(null, true); return false;
          });
        }, more: function (v) { return v !== false; }});
        poll.start();
      });
    });
  });
  form.querySelectorAll("[data-remove-folder]").forEach(function (b) {
    b.addEventListener("click", function () {
      b.disabled = true;
      PB.post("/v1/benchmark/datasets/remove", new URLSearchParams({dataset: b.dataset.removeFolder})).then(function (j) {
        say(j.message || (j.success ? "Removed." : "Not removed."), j.success ? "ok" : "error");
        if (j.success) window.PrismLive.reloadKeep(); else b.disabled = false;
      });
    });
  });

  // ---- the run's combos ------------------------------------------------------------------------------
  var live = $("run-live"), list = $("run-combos"), lastKey = "";
  var CHIP = {done: "ok", failed: "bad", skipped: "same", queued: "same"};
  S.onPoll(function (st) {
    var items = BC.comboStates(st.run), show = st.running || (st.ended && items.length);
    live.hidden = !items.length || (!show && !(st.state === "done" || st.state === "failed" || st.state === "cancelled"));
    start.disabled = !!st.running;
    start.textContent = st.running ? "Running…" : "Start benchmark";
    $("run-done").hidden = st.running || !(st.state === "done" || st.state === "cancelled" || st.state === "failed");
    $("run-live-note").textContent = st.running ? "Compare keeps showing the previous results until this run ends." : (st.run && st.run.error ? "The run stopped: " + st.run.error : st.run && st.run.message ? st.run.message : "");
    var key = JSON.stringify(items);
    if (key === lastKey) return;
    lastKey = key;
    list.replaceChildren();
    items.forEach(function (it) {
      list.appendChild(el("li", {class: it.now ? "now" : null}, [
        el("span", {}, [el("strong", {text: it.model}), " on ", el("code", {text: it.device || "its device"})]),
        el("span", {class: "bn-chip " + (CHIP[it.state] || "warn"), text: it.text})
      ]));
    });
  });

  deviceRules(); estimate();
})();
