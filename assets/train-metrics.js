// Blue Onyx Prism: Training page metrics view (window.TrainMetrics).
// render(el, progress, columns, metrics, opts) fills one `.tr-metrics` block from a run's
// progress record (`history` per epoch, `best`): KPI tiles with the change since the previous
// epoch (learning rate, convergence countdown, finish time among them), an overfitting hint,
// loss, quality and learning-rate curves (PrismBench.charts, legend toggles, "Show as table"), a
// per-epoch table over the chosen columns and the columns picker. refresh(el, progress, metrics,
// opts) swaps only the tiles and the hint (every poll; render() when signature() changes).
// opts: runId, available [keys the picker offers], onColumns(list) -> Promise<{ok, message}>.
// Everything is built with createElement / textContent (metric keys come from a training
// script); colours come from CSS classes (CSP style-src 'self'). ES5, no build step.
(function () {
  "use strict";
  var PB = window.PrismBench || {};
  var has = function (v) { return v !== null && v !== undefined && typeof v === "number" && !isNaN(v); };
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  // UI state that survives re-renders (the block is rebuilt when the history grows).
  var state = { all: {}, cols: {}, tbl: {}, more: {} };

  // ---- number formats --------------------------------------------------------------------------
  function fixedGuess(v) { var a = Math.abs(v); return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : v.toFixed(3); }
  function format(kind, v) {
    if (!has(v)) return "-";
    switch (kind) {
      case "fixed3": return v.toFixed(3);
      case "sci": return v.toExponential(2);
      case "pct": return (v * 100).toFixed(1) + "%";
      case "int": return String(Math.round(v));
      case "gb": return v.toFixed(1) + " GB";
      case "rate": return v.toFixed(1) + "/s";
      default: return fixedGuess(v);
    }
  }
  function duration(s) {
    if (!has(s) || s < 0) return "-";
    s = Math.round(s);
    if (s < 90) return s + " s";
    var m = Math.round(s / 60);
    if (m < 90) return m + " min";
    return (m / 60).toFixed(1) + " h";
  }
  // Shortest scientific text of a learning rate: 1e-4, 1.5e-4 (an f32 widened to f64 such as
  // 4.999999873689376e-5 reads 5e-5).
  function shortSci(v) {
    if (!has(v)) return "-";
    return v === 0 ? "0" : (+v.toPrecision(4)).toExponential();
  }
  // Wall-clock time `eta_s` seconds from now ("14:20"; "Sat 09:05" when over 20 h away).
  function doneAt(etaS) {
    var d = new Date(Date.now() + etaS * 1000);
    var t = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return etaS > 20 * 3600 ? d.toLocaleDateString([], { weekday: "short" }) + " " + t : t;
  }

  // id -> {id, label, short, hb, format} from the server's list; raw keys describe themselves.
  function infoMap(metrics) {
    var m = {};
    (metrics || []).forEach(function (d) {
      if (d && d.id) m[d.id] = { id: d.id, label: d.label || d.id, short: d.short || d.label || d.id, hb: d.higher_better === true ? true : d.higher_better === false ? false : null, format: d.format || null };
    });
    return function (id) {
      return m[id] || { id: id, label: id, short: id, hb: /loss/i.test(id) ? false : null, format: null };
    };
  }

  // ---- history helpers -------------------------------------------------------------------------
  function keysOf(history) {
    var seen = {}, out = [];
    history.forEach(function (h) {
      Object.keys(h.metrics || {}).forEach(function (k) { if (!seen[k] && has(h.metrics[k])) { seen[k] = true; out.push(k); } });
    });
    return out;
  }
  function seriesOf(history, key) {
    return history.map(function (h) { return { x: h.epoch, y: has((h.metrics || {})[key]) ? h.metrics[key] : null }; });
  }
  function lastTwo(history, key) {
    var vals = [];
    history.forEach(function (h) { var v = (h.metrics || {})[key]; if (has(v)) vals.push(v); });
    return { last: vals.length ? vals[vals.length - 1] : null, prev: vals.length > 1 ? vals[vals.length - 2] : null };
  }
  function peak(history, key) {
    var best = null;
    history.forEach(function (h) { var v = (h.metrics || {})[key]; if (has(v) && (best === null || v > best)) best = v; });
    return best;
  }

  // ---- KPI tiles -------------------------------------------------------------------------------
  function deltaNode(info, last, prev) {
    if (!has(last) || !has(prev)) return null;
    var d = last - prev;
    var shown = info.format === "pct" ? Math.abs(d * 100).toFixed(1) + " points" : format(info.format, Math.abs(d));
    if (d === 0) return el("span", "tr-delta same", "no change");
    var good = info.hb === null ? null : (d > 0) === info.hb;
    return el("span", "tr-delta " + (good === null ? "neutral" : good ? "good" : "bad"), (d > 0 ? "▲ " : "▼ ") + shown);
  }
  function tile(label, value, sub, delta) {
    var t = el("div", "tr-kpi");
    t.appendChild(el("div", "tr-kpi-k", label));
    t.appendChild(el("div", "tr-kpi-v", value));
    if (sub) t.appendChild(el("div", "tr-kpi-s muted", sub));
    if (delta) t.appendChild(delta);
    return t;
  }
  // The convergence rule's countdown: epochs since the last gain against the patience, where the
  // run stops (the server's `stop_epoch`) and the extension / anneal it is in. Null when early
  // stopping is off or the run predates it (patience 0 or missing).
  function convergenceTile(progress) {
    if (!has(progress.patience) || progress.patience <= 0 || !has(progress.since_best)) return null;
    var subs = [], st = progress.stopped, an = progress.anneal, cp = progress.converge_point;
    if (st && has(st.epoch)) subs.push("stopped at epoch " + st.epoch + (st.reason ? " (" + st.reason + ")" : ""));
    else if (an && has(an.until)) subs.push("annealing (lower learning rate), stops at epoch " + an.until);
    else {
      if (has(progress.stop_epoch)) subs.push("stops at epoch " + progress.stop_epoch);
      if (cp && has(cp.extension) && cp.extension > 0) subs.push("extension " + cp.extension + (has(cp.next_check) ? ", next check epoch " + cp.next_check : ""));
    }
    return tile("Convergence", "since best " + progress.since_best + " / " + progress.patience, subs.join("; "));
  }
  // The current learning rate (the last batch's when the trainer reports it, else the last
  // epoch's) with the change against the previous epoch's; direction is neither good nor bad.
  function lrTile(progress, history) {
    var lt = lastTwo(history, "lr");
    var cur = has(progress.lr) ? progress.lr : lt.last;
    if (!has(cur)) return null;
    // The epoch event sets progress.lr to the epoch's value: compare with the one before it.
    var prev = has(progress.lr) && cur !== lt.last ? lt.last : lt.prev;
    return tile("Learning rate", format("sci", cur), "", deltaNode({ hb: null, format: "sci" }, cur, prev));
  }
  function tiles(progress, history, info, running) {
    var box = el("div", "tr-kpis");
    if (has(progress.epoch)) {
      var sub = has(progress.batch) && has(progress.batches) && progress.batches > 0 ? "batch " + progress.batch + "/" + progress.batches : "";
      box.appendChild(tile("Epoch", progress.epoch + (has(progress.epochs) ? " / " + progress.epochs : ""), sub));
    }
    if (running && has(progress.eta_s)) {
      var so = has(progress.elapsed_s) ? duration(progress.elapsed_s) + " so far" : "";
      box.appendChild(tile("Time left", "~" + duration(progress.eta_s), (so ? so + " · " : "") + "done ~" + doneAt(progress.eta_s)));
    } else if (has(progress.elapsed_s)) box.appendChild(tile("Time", duration(progress.elapsed_s), ""));
    var conv = convergenceTile(progress);
    if (conv) box.appendChild(conv);
    var best = progress.best;
    if (best && has(best.value)) {
      var bi = info(best.metric);
      box.appendChild(tile("Best " + bi.short, format(bi.format, best.value), "epoch " + best.epoch));
    }
    [["loss", "Loss"], ["val_loss", "Val loss"]].forEach(function (p) {
      var lt = lastTwo(history, p[0]);
      if (!has(lt.last)) return;
      var i = info(p[0]);
      box.appendChild(tile(p[1], format(i.format || "fixed3", lt.last), "", deltaNode({ hb: i.hb === null ? false : i.hb, format: i.format }, lt.last, lt.prev)));
    });
    var lr = lrTile(progress, history);
    if (lr) box.appendChild(lr);
    [["mem_gb", "Peak memory", "gb"], ["img_s", "Peak speed", "rate"]].forEach(function (p) {
      var v = peak(history, p[0]);
      if (!has(v)) return;
      var i = info(p[0]);
      box.appendChild(tile(p[1], format(i.format || p[2], v), ""));
    });
    return box;
  }

  // ---- overfitting hint ------------------------------------------------------------------------
  // Pure: true when the mean validation loss of the last 3 epochs is above that of the 3 epochs
  // before them while the mean training loss fell over the same windows, and the convergence
  // rule has seen no gain for max(3, patience / 2) epochs (a rising validation loss alone is
  // common early on). Epochs missing either loss are left out; fewer than 6 such epochs, or a
  // family that reports no validation loss, is never a hint.
  var OVERFIT_WINDOW = 3;
  function overfitting(history, progress) {
    progress = progress || {};
    var rows = (history || []).filter(function (h) { var m = h.metrics || {}; return has(m.loss) && has(m.val_loss); });
    var n = OVERFIT_WINDOW;
    if (rows.length < 2 * n) return false;
    var mean = function (list, key) { var s = 0; list.forEach(function (h) { s += h.metrics[key]; }); return s / list.length; };
    var now = rows.slice(-n), before = rows.slice(-2 * n, -n);
    var valRose = mean(now, "val_loss") > mean(before, "val_loss");
    var lossFell = mean(now, "loss") < mean(before, "loss");
    var patience = has(progress.patience) ? progress.patience : 0;
    var stale = has(progress.since_best) && progress.since_best >= Math.max(3, patience / 2);
    return valRose && lossFell && stale;
  }
  function overfitNote(history, progress) {
    if (!overfitting(history, progress)) return null;
    var b = progress.best && has(progress.best.epoch) ? " (" + progress.best.epoch + ")" : "";
    return el("p", "warn small tr-overfit", "Validation loss rising while training loss falls: likely overfitting; the best epoch" + b + " is kept.");
  }
  // The parts that change with every batch (tiles, overfitting hint): refresh() swaps them in
  // place so the charts and table are rebuilt only when the history or the columns change.
  function live(progress, history, info, running) {
    var box = el("div", "tr-live");
    box.appendChild(tiles(progress, history, info, running));
    var w = overfitNote(history, progress);
    if (w) box.appendChild(w);
    return box;
  }

  // ---- charts ----------------------------------------------------------------------------------
  var QUALITY = /^(ema_)?(map|map50|map75|mar|precision|recall|f1|f2|f0_5)$/;
  function intTick(v) { return String(Math.round(v)); }
  // valFmt (optional): the hover and table text of a value (default: percent with pct, else 4
  // significant digits).
  function chart(runId, kind, title, keys, history, info, progress, yFmt, pct, valFmt) {
    var C = PB.charts;
    if (!C || !keys.length) return null;
    var val = valFmt || (pct ? function (v) { return (v * 100).toFixed(2) + "%"; } : function (v) { return String(+v.toPrecision(4)); });
    var series = keys.map(function (k, i) {
      return { id: k, label: info(k).label, cls: "ser ser-" + ((i % 8) + 1), dash: C.dashOf(i), points: seriesOf(history, k) };
    });
    var xs = history.map(function (h) { return h.epoch; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    if (x1 === x0) x1 = x0 + 1;
    var ymax = 1;
    if (!pct) {
      var m = 0;
      series.forEach(function (s) { s.points.forEach(function (p) { if (has(p.y) && p.y > m) m = p.y; }); });
      ymax = C.niceMax(m || 1);
    }
    var best = progress.best, marks = best && has(best.epoch) ? [{ x: best.epoch, cls: "thr-mark-best", label: "best " + best.epoch }] : [];
    var ticks = C.niceTicks(x0, x1, 6).filter(function (v) { return v === Math.round(v); });
    var legend = series.map(function (s) { return { id: s.id, label: s.label, cls: s.cls, dash: s.dash }; });
    var blk = C.block({
      key: "tm:" + runId + ":" + kind, series: legend,
      draw: function (hidden) {
        return C.lineChart(series, {
          title: title, desc: title + " by epoch", hidden: hidden, xMin: x0, xMax: x1, yMin: 0, yMax: ymax, xTicks: ticks, xFmt: intTick,
          yFmt: yFmt, yHover: val,
          xName: "epoch", xLabel: "Epoch", yLabel: pct ? "%" : "", marks: marks, pointDots: history.length < 40, hover: "x", left: pct ? 44 : 56
        });
      },
      table: function () {
        return { head: ["Epoch"].concat(series.map(function (s) { return s.label; })), rows: history.map(function (h, i) {
          return [String(h.epoch)].concat(series.map(function (s) { var y = s.points[i].y; return has(y) ? val(y) : "-"; }));
        }) };
      }
    });
    blk.classList.add("tr-chart-block");
    var cap = el("h4", "tr-chart-title", title);
    var wrap = el("div", "tr-chart-wrap");
    wrap.appendChild(cap); wrap.appendChild(blk);
    var d = blk.querySelector("details.chart-table"), tk = runId + ":" + kind;
    if (d) {
      if (state.tbl[tk]) d.open = true;
      d.addEventListener("toggle", function () { state.tbl[tk] = d.open; });
    }
    return wrap;
  }

  // ---- per-epoch table -------------------------------------------------------------------------
  function table(runId, history, cols, info, progress) {
    var present = keysOf(history), shownCols = cols.filter(function (c) { return present.indexOf(c) >= 0; });
    if (!shownCols.length) return null;
    var bestEpoch = progress.best && has(progress.best.epoch) ? progress.best.epoch : null;
    var bestOf = {};
    shownCols.forEach(function (c) {
      var i = info(c);
      if (i.hb === null) return;
      var b = null;
      history.forEach(function (h) { var v = (h.metrics || {})[c]; if (has(v) && (b === null || (i.hb ? v > b : v < b))) b = v; });
      bestOf[c] = b;
    });
    var rows = history.slice().reverse();
    var all = !!state.all[runId], limit = 10;
    var shown = all ? rows : rows.slice(0, limit);
    var wrap = el("div", "tr-table-wrap");
    var sc = el("div", "scroll");
    var t = el("table", "tr-mtable"), thead = el("thead"), hr = el("tr");
    hr.appendChild(el("th", null, "Epoch"));
    shownCols.forEach(function (c) { var th = el("th", "num", info(c).short); th.title = info(c).label; hr.appendChild(th); });
    thead.appendChild(hr); t.appendChild(thead);
    var tb = el("tbody");
    shown.forEach(function (h) {
      var tr = el("tr", h.epoch === bestEpoch ? "best-epoch" : null);
      tr.appendChild(el("td", null, String(h.epoch) + (h.epoch === bestEpoch ? " ★" : "")));
      shownCols.forEach(function (c) {
        var v = (h.metrics || {})[c], i = info(c);
        var td = el("td", "num" + (has(v) && bestOf[c] === v ? " best-val" : ""), format(i.format, v));
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb); sc.appendChild(t); wrap.appendChild(sc);
    if (rows.length > limit) {
      var b = el("button", "small secondary", all ? "Show the last " + limit + " epochs" : "Show all " + rows.length + " epochs");
      b.type = "button";
      b.addEventListener("click", function () { state.all[runId] = !all; wrap.parentNode.replaceChild(table(runId, history, cols, info, progress), wrap); });
      wrap.appendChild(b);
    }
    return wrap;
  }

  // ---- columns picker --------------------------------------------------------------------------
  function picker(runId, cols, available, info, onColumns) {
    var d = el("details", "tr-cols");
    d.appendChild(el("summary", null, "Columns"));
    if (state.cols[runId]) d.open = true;
    d.addEventListener("toggle", function () { state.cols[runId] = d.open; });
    var chosen = cols.filter(function (c) { return available.indexOf(c) >= 0; });
    var rest = available.filter(function (k) { return chosen.indexOf(k) < 0; });
    var msg = el("p", "error-text small tr-cols-msg");
    msg.setAttribute("role", "status");
    function push(list) {
      msg.textContent = "";
      if (!onColumns) return;
      onColumns(list).then(function (r) { if (r && r.ok === false) msg.textContent = r.message || "The server did not accept the columns."; }, function (e) { msg.textContent = (e && e.message) || "The columns could not be saved."; });
    }
    // Raw keys the service has no label for (per-term losses, learning-rate bounds) go under
    // "More" so the known metrics stay in view.
    var isRaw = function (k) { var i = info(k); return i.label === k && !i.format; };
    var list = el("ul", "tr-cols-list"), more = el("ul", "tr-cols-list"), moreN = 0;
    chosen.concat(rest).forEach(function (k, idx) {
      var isOn = chosen.indexOf(k) >= 0, li = el("li");
      var lab = el("label", "check"), cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = isOn;
      cb.addEventListener("change", function () {
        var next = chosen.slice();
        if (cb.checked) next.push(k); else next.splice(next.indexOf(k), 1);
        push(next);
      });
      lab.appendChild(cb); lab.appendChild(document.createTextNode(" " + info(k).label));
      li.appendChild(lab);
      if (info(k).label !== k) li.appendChild(el("code", "muted small", " " + k));
      if (isOn) {
        var pos = chosen.indexOf(k);
        [["↑", -1, "Move " + info(k).label + " up"], ["↓", 1, "Move " + info(k).label + " down"]].forEach(function (a) {
          var b = el("button", "small secondary", a[0]);
          b.type = "button"; b.title = a[2]; b.setAttribute("aria-label", a[2]);
          b.disabled = pos + a[1] < 0 || pos + a[1] >= chosen.length;
          b.addEventListener("click", function () {
            var next = chosen.slice(), o = next[pos + a[1]];
            next[pos + a[1]] = next[pos]; next[pos] = o;
            push(next);
          });
          li.appendChild(b);
        });
      }
      if (!isOn && isRaw(k)) { more.appendChild(li); moreN++; } else list.appendChild(li);
    });
    d.appendChild(list);
    if (moreN) {
      var md = el("details", "tr-cols-more");
      md.appendChild(el("summary", "small", "More (" + moreN + " keys reported by the trainer)"));
      if (state.more[runId]) md.open = true;
      md.addEventListener("toggle", function () { state.more[runId] = md.open; });
      md.appendChild(more); d.appendChild(md);
    }
    d.appendChild(msg);
    return d;
  }

  // ---- entry points ----------------------------------------------------------------------------
  // The reasons to rebuild a block: the history grew, the epoch or best moved or the columns
  // changed. The batch, the per-batch learning rate and the time left are left out: they change
  // on every poll and only touch the tiles, which refresh() swaps in place (a rebuild would redraw
  // the charts every 3 s and drop a hover).
  function signature(progress, columns) {
    if (!progress) return "";
    var n = progress.history ? progress.history.length : 0;
    return [n, progress.epoch, progress.best ? progress.best.epoch + ":" + progress.best.value : "", (columns || []).join(",")].join("|");
  }
  // Something to show before the first epoch ends (the tiles only): an epoch under way or a
  // learning rate reported.
  function hasLive(progress) {
    return !!progress && ((has(progress.epoch) && progress.epoch > 0) || has(progress.lr));
  }

  // Swap the tiles and the overfitting hint of a block render() built; false when there is no
  // such block (the caller renders instead).
  function refresh(host, progress, metrics, opts) {
    opts = opts || {};
    var old = host.querySelector(".tr-live");
    if (!old || !progress) return false;
    var history = progress.history || [];
    old.parentNode.replaceChild(live(progress, history, infoMap(metrics), !!opts.running), old);
    return true;
  }

  function render(host, progress, columns, metrics, opts) {
    opts = opts || {};
    var history = progress && progress.history && progress.history.length ? progress.history : null;
    host.replaceChildren();
    var info = infoMap(metrics), running = !!opts.running;
    if (!history) {
      // Epoch 1 under way: the tiles (epoch, time left, learning rate in warm-up) without curves.
      if (!hasLive(progress)) return false;
      host.appendChild(live(progress, [], info, running));
      return true;
    }
    var runId = opts.runId || host.dataset.run || "", cols = columns || [];
    var keys = keysOf(history);
    host.appendChild(live(progress, history, info, running));

    // The charts follow the chosen columns (training / validation loss always), so a run with
    // dozens of raw keys still draws a few readable series; one epoch is no curve.
    var lossKeys = keys.filter(function (k) { return (k === "loss" || k === "val_loss" || cols.indexOf(k) >= 0) && /loss/i.test(k); });
    var isQuality = function (k) { return QUALITY.test(k) || (info(k).format === "pct" && !/loss/i.test(k)); };
    var qualityKeys = cols.filter(function (k) { return keys.indexOf(k) >= 0 && isQuality(k); });
    if (!qualityKeys.length) qualityKeys = ["map", "map50"].filter(function (k) { return keys.indexOf(k) >= 0; });
    if (history.length < 2) { lossKeys = []; qualityKeys = []; }
    var charts = el("div", "tr-charts");
    var c1 = chart(runId, "loss", "Loss", lossKeys, history, info, progress, function (v) { return String(+v.toPrecision(3)); }, false);
    var c2 = chart(runId, "quality", "Quality (higher is better)", qualityKeys, history, info, progress, function (v) { return (v * 100).toFixed(0); }, true);
    // Learning rate on its own axis (1e-4 against a loss of 3 would be a flat line at zero): the
    // axis still starts at 0 and niceMax rounds tiny maxima fine (1e-4 -> ticks 0, 2e-5 .. 1e-4),
    // the tick text is scientific so 0.00002 does not crowd the margin.
    var lrKeys = history.length >= 2 && keys.indexOf("lr") >= 0 ? ["lr"] : [];
    var c3 = chart(runId, "lr", "Learning rate", lrKeys, history, info, progress, shortSci, false, function (v) { return format("sci", v); });
    if (c1) charts.appendChild(c1);
    if (c2) charts.appendChild(c2);
    if (c3) charts.appendChild(c3);
    if (charts.childNodes.length) host.appendChild(charts);

    var tw = table(runId, history, cols, info, progress);
    if (tw) host.appendChild(tw);
    var available = opts.available && opts.available.length ? opts.available : keys;
    host.appendChild(picker(runId, cols, available, info, opts.onColumns));
    return true;
  }

  window.TrainMetrics = {
    render: render, refresh: refresh, signature: signature, format: format, infoMap: infoMap, keysOf: keysOf,
    overfitting: overfitting, shortSci: shortSci
  };
})();
