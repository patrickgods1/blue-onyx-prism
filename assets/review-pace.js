// Review tab, session pace (/train/review; loaded before review.js): the session line under the
// message line ("Session 23 frames · 64 boxes, 2.8 a minute · 82% accepted · 37 min"), the HUD's
// pace slot and Details' copy of the line, a "Going fast" chip when the last 10 frames went much
// faster while accepting more (rubber-stamping), and a break note every 45 min. The maths
// (`median`, `acceptRate`, `boxesPerMinute`, `goingFast`, `frameRecord`, `sessionText`) is pure
// and runs in Node (tests/it/js_review_pace.rs); `init` wires the page part with what review.js
// passes in and keeps the session (page memory only: a reload starts a new one).
(function () {
  'use strict';

  function median(a) {
    var s = a.slice().sort(function (x, y) { return x - y; }), m = s.length >> 1;
    return s.length ? (s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2) : 0;
  }

  function acceptRate(fs) {
    var a = 0, r = 0;
    fs.forEach(function (f) { a += f.acc; r += f.rej; });
    return a + r ? a / (a + r) : null;
  }

  function seconds(fs) { return fs.map(function (f) { return f.s; }); }

  // A saved frame: its time (load to save, `ms`, at most 10 min) and the reviewer's box decisions
  // on it (`R.boxDecisions`: boxes pending when served and now decided, plus boxes drawn; a
  // duplicate, a too-small box, a dismissal or a parked accept is no verdict on an object).
  function frameRecord(d, ms) {
    return { s: Math.min(600, ms / 1000), acc: d.accepted, rej: d.rejected, boxes: d.decided + d.drawn };
  }

  // Boxes a minute of review: the frames' own (capped) times, so a break or a frame left open
  // does not count; '' before a second of it.
  function boxesPerMinute(fs) {
    var n = 0, s = 0;
    fs.forEach(function (f) { n += f.boxes || 0; s += f.s; });
    if (s < 1) return '';
    var r = n * 60 / s;
    return r >= 10 ? String(Math.round(r)) : r.toFixed(1);
  }

  // From 15 frames on: the last 10 went much faster (median time under 0.4 of the session's)
  // while accepting more than the session does.
  function goingFast(fs) {
    if (fs.length < 15) return false;
    var rate = acceptRate(fs), last = fs.slice(-10), lr = acceptRate(last);
    return median(seconds(last)) < 0.4 * median(seconds(fs)) && lr != null && rate != null && lr > rate;
  }

  // The words of a session of saved frames `fs` (not empty), `min` minutes long: `hud`, the HUD's
  // pace slot ("64 boxes · 2.8/min"), and `line`, the session line (effort in boxes: a frame of
  // one box and one of twelve are not the same work).
  function sessionText(fs, min) {
    var rate = acceptRate(fs);
    var nb = 0;
    fs.forEach(function (f) { nb += f.boxes || 0; });
    var pace = boxesPerMinute(fs);
    var parts = [
      'Session ' + fs.length + ' frame' + (fs.length === 1 ? '' : 's'),
      nb + ' box' + (nb === 1 ? '' : 'es') + (nb && pace ? ', ' + pace + ' a minute' : '')
    ];
    if (rate != null) parts.push(Math.round(rate * 100) + '% accepted');
    parts.push(min + ' min');
    return {
      hud: nb + ' box' + (nb === 1 ? '' : 'es') + (nb && pace ? ' · ' + pace + '/min' : ''),
      line: parts.join(' · ')
    };
  }

  // The page part. `c`: `el`, `button` (dom.js / review.js), `sessionLine` (#rv-session, may be
  // null: then nothing shows), `detailsPace` (Details' copy, may be null), `hudSet(element, text)`
  // and `hudPace` (the HUD's pace slot). Returns `noteSaved(decisions, frameStart)` (a frame
  // saved as reviewed, loaded at `frameStart`) and `show` (draw the line again).
  function init(c) {
    var el = c.el, button = c.button, sessionLine = c.sessionLine, detailsPace = c.detailsPace;
    var hudSet = c.hudSet, hudPace = c.hudPace;
    var session = {
      start: Date.now(), frames: [],     // per saved frame {s, acc, rej, boxes}
      breakAt: Date.now() + 45 * 60000   // the next break note
    };

    function noteSaved(d, frameStart) {
      session.frames.push(frameRecord(d, Date.now() - frameStart));
      showSession();
    }

    // The session line, a note when the last 10 frames went much faster while accepting more,
    // and a break note every 45 min.
    function showSession() {
      if (!sessionLine) return;
      var fs = session.frames, min = Math.floor((Date.now() - session.start) / 60000);
      sessionLine.textContent = '';
      if (!fs.length) {
        hudSet(hudPace, '');
        if (detailsPace) detailsPace.textContent = '';
        return;
      }
      var text = sessionText(fs, min);
      // The HUD's pace slot (Details holds the whole line where the HUD has no room for it).
      hudSet(hudPace, text.hud);
      // The plain stats (two columns hide them: the HUD's pace slot and Details have them; the
      // warning chips after them stay).
      sessionLine.appendChild(el('span', 'rv-session-stats', text.line));
      if (detailsPace) detailsPace.textContent = text.line;
      if (goingFast(fs)) {
        sessionLine.appendChild(el('span', 'rv-chip warn', 'Going fast: check the small boxes'));
      }
      if (Date.now() >= session.breakAt) {
        var note = el('span', 'rv-chip warn', 'Take a break? ');
        note.appendChild(button('Later', function () { session.breakAt = Date.now() + 45 * 60000; showSession(); }, 'secondary'));
        sessionLine.appendChild(note);
      }
    }
    setInterval(showSession, 30000);

    return { noteSaved: noteSaved, show: showSession };
  }

  globalThis.ReviewPace = {
    median: median, acceptRate: acceptRate, seconds: seconds, frameRecord: frameRecord,
    boxesPerMinute: boxesPerMinute, goingFast: goingFast, sessionText: sessionText, init: init
  };
})();
