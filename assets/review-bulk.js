// Review tab, following a job and the Bulk tools (/train/review; loaded before review.js). A job
// queued on the Auto-label tab (or the overlap check queued here) is followed on
// GET /v1/train/jobs?task=<id> (its record only), every 3 s while the tab shows (PrismPoll), and
// the frames reload when it ends. Under Bulk tools, "Accept auto: accept frames" calls
// POST /v1/review/accept-auto and "Reject auto boxes below" POST /v1/review/reject-low, "Reject
// boxes of unwanted classes" POST /v1/review/reject-other-class, "Restore reopened frames"
// POST /v1/review/restore-reopened and "Reject stacked duplicates" POST /v1/review/reject-stacked,
// plus accept-repeats and suggest-overlaps (each one bulkAction(): dry run, confirm, run, report).
// *Suggest overlaps...* queues the overlap check (POST /v1/review/suggest-overlaps, followed like
// an auto-label job). `describeTask` is pure; `init` wires the page part with what review.js
// passes in. tests/it/js_review_bulk.rs runs this file in Node on a fake page.
(function () {
  'use strict';

  // The line a followed task shows while it is queued, running or paused (its record from
  // GET /v1/train/jobs).
  function describeTask(t) {
    if (t.state === 'queued') return 'Queued: ' + (t.title || 'auto-label') + ' (waiting for the queue).';
    var s = t.stage ? String(t.stage).replace(/_/g, ' ') : 'starting';
    return (t.title ? t.title + ': ' : '') + s + (t.steps ? ' (' + t.step + '/' + t.steps + ')' : '') +
      (t.percent != null ? ', ' + t.percent + '%' : '') + (t.detail ? ': ' + t.detail : '');
  }

  // Wires the job line and the bulk buttons, and follows a job queued earlier. `c`: `$` (an
  // element by id), `el`, `report`, `postJson` (dom.js), `button` and `msg` (review.js's message
  // line), `filtersForm` (the filters each action sends), `reloadQueue` (after a job or a bulk
  // change: unsaved edits stay), `showOverlaps` and `filterReload` (the result's "Show the
  // overlapping frames"), `forgetHistories` (rejected stacks were decisions of these cameras).
  // Returns `bulkAction` and `pollTask` (for the tests).
  function init(c) {
    var $ = c.$, el = c.el, report = c.report, postJson = c.postJson, button = c.button, msg = c.msg;
    var filtersForm = c.filtersForm, reloadQueue = c.reloadQueue;
    var showOverlaps = c.showOverlaps, filterReload = c.filterReload;

    var alStatus = $('rv-al-status');
    var alProgress = $('rv-al-progress');
    var polling = null;   // {id, box, bar, after, loop}: the task followed

    function stopPolling() {
      if (polling) polling.loop.stop();
      polling = null;
    }

    // Task records: task `id` (a list of one, none when the server has no such task), else every
    // queued, running or paused one. From GET /v1/train/jobs (task=<id> / tasks=busy: the records
    // without log or result). The server always sends the field (null for an unknown task); an
    // answer without it is an error, never a reason to ask another route.
    function taskRecords(id) {
      // Asked from a poll: a poll throughout.
      var get = PrismPoll.inPoll() ? PrismPoll.fetch : fetch;
      var json = function (url) {
        return get(url, { headers: { Accept: 'application/json' }, cache: 'no-store' }).then(function (r) {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        });
      };
      var key = id ? 'task' : 'tasks';
      return json('/v1/train/jobs?' + (id ? 'task=' + encodeURIComponent(id) : 'tasks=busy')).then(function (j) {
        if (!j || typeof j !== 'object' || !(key in j)) throw new Error('the jobs answer has no ' + key + ' field');
        return id ? (j.task ? [j.task] : []) : (j.tasks || []);
      });
    }

    // Chained, never an interval (no two requests at once), no request while the tab is hidden, one
    // at once when it shows again. `run()` resolves false to stop.
    function loop(run, period) {
      var p = PrismPoll.create({ run: run, period: period, more: function (go) { return go !== false; } });
      p.start();
      return p;
    }

    // Follow task `id`; `box` and `bar` show it (default: the line above the frame), `after(t)` runs when
    // it ends.
    function pollTask(id, box, bar, after) {
      stopPolling();
      var me = { id: id, box: box || alStatus, bar: bar || alProgress, after: after };
      polling = me;
      me.loop = loop(function () { return tick(me); }, 3000);
    }

    function tick(me) {
      return taskRecords(me.id).then(function (list) {
        if (polling !== me) return false;
        var t = list.filter(function (x) { return x.id === me.id; })[0];
        if (!t) return true;
        var box = me.box, bar = me.bar, after = me.after;
        if (t.state === 'queued' || t.state === 'running' || t.state === 'paused') {
          report(box, [describeTask(t)]);
          bar.hidden = false;
          bar.value = t.percent || 0;
          return true;
        }
        stopPolling();
        bar.hidden = true;
        if (t.state === 'done') report(box, ['Done: ' + (t.summary || t.title || 'auto-label finished')]);
        else if (t.state === 'failed') report(box, ['Failed: ' + (t.error || 'see Jobs on the Auto-label tab.')]);
        else report(box, [t.summary ? 'Cancelled: ' + t.summary : 'Cancelled.']);
        box.className = 'review-report ' + (t.state === 'done' ? 'ok-text' : 'error-text');
        // Asked for by the poll's answer, not the reviewer.
        PrismPoll.background(function () {
          if (after) after(t);
          reloadQueue();
        });
        return false;
      });
    }

    // A job queued earlier (here or on the Auto-label tab): follow it.
    if (alStatus) {
      taskRecords(null)
        .then(function (list) {
          if (polling) return;
          var busyTasks = list.filter(function (t) {
            return (t.kind === 'autolabel' || t.kind === 'overlaps') && (t.state === 'queued' || t.state === 'running' || t.state === 'paused');
          });
          var t = busyTasks.filter(function (x) { return x.state === 'running'; })[0] || busyTasks[0];
          if (t && t.kind === 'overlaps') followOverlaps(t.id);
          else if (t) pollTask(t.id);
        })
        .catch(function () {});
    }

    // The seven bulk buttons share one flow: a dry run of `o.url` ("Counting..."), nothing to do
    // or a confirm, the real run, its report and a reload. `o`:
    //   button, box   the button (disabled meanwhile) and the report box
    //   url           POST /v1/review/<action>; body = `start` (default {}), then the filters named
    //                 by `keys`, then `extra(body)`, then `dry_run`
    //   keys          the filter fields sent. Required per action: SuggestOverlapsBody and the
    //                 stacked walk take no `source` (SuggestOverlapsBody rejects unknown fields, so
    //                 sending it would answer 400)
    //   counting      the first line, while the dry run runs
    //   count(j)      what the dry run found; 0 = nothing to do
    //   none(j)       the report lines for nothing (afterNone(j), when given, runs next: a promise)
    //   gate(j, body) optional, after "something found": true = stop (it has reported)
    //   confirm(j, body, n)  the confirm text; cancelled: the line when the user says no
    //   done(r)       the first report line of the real run; after(r) runs next (a promise)
    //   finish(r)     replaces done/after (the overlap check queues a job and has no outcome colour)
    // `failed` entries are strings (Vec<String> in bulk.rs); any other shape prints as JSON.
    function bulkAction(o) {
      var box = o.box, body = o.start || {};
      o.keys.forEach(function (k) {
        var e = filtersForm.elements[k];
        if (e && e.value) body[k] = e.value;
      });
      if (o.extra) o.extra(body);
      body.dry_run = true;
      box.className = 'review-report';
      report(box, [o.counting]);
      o.button.disabled = true;
      return postJson(o.url, body).then(function (j) {
        var n = o.count(j);
        if (!n) {
          report(box, o.none(j));
          return o.afterNone ? o.afterNone(j) : null;
        }
        if (o.gate && o.gate(j, body)) return null;
        if (!window.confirm(o.confirm(j, body, n))) {
          report(box, [o.cancelled]);
          return null;
        }
        body.dry_run = false;
        return postJson(o.url, body).then(function (r) {
          if (o.finish) return o.finish(r);
          var failed = r.failed || [];
          report(box, [o.done(r)],
            failed.map(function (f) { return typeof f === 'string' ? f : JSON.stringify(f); }));
          box.className = 'review-report ' + (failed.length ? 'error-text' : 'ok-text');
          return o.after(r);
        });
      }).catch(function (e) {
        box.className = 'review-report error-text';
        report(box, [e.message]);
      }).then(function () { o.button.disabled = false; });
    }

    var ALL_KEYS = ['bucket', 'scene', 'camera', 'label', 'source'];
    var NO_SOURCE_KEYS = ['bucket', 'scene', 'camera', 'label'];

    var bulkBtn = $('rv-accept-auto');
    if (bulkBtn) {
      bulkBtn.addEventListener('click', function () {
        bulkAction({
          button: bulkBtn, box: $('rv-accept-auto-msg'), url: '/v1/review/accept-auto', keys: ALL_KEYS,
          counting: 'Counting the frames...',
          count: function (j) { return j.eligible || 0; },
          none: function (j) { return [j.message || 'No unreviewed frame with only auto: accept boxes matches the filters.']; },
          confirm: function (j, body, n) {
            return 'Accept ' + n + ' frame(s) whose boxes the auto-labeler marked accept? ' +
              'They are saved as reviewed: those boxes accepted, unconfirmed hint boxes rejected. ' +
              'Frames with a box flagged for review are left alone.';
          },
          cancelled: 'Nothing accepted.',
          done: function (r) { return r.message || ('Accepted ' + (r.accepted || 0) + ' frame(s).'); },
          after: function () { return reloadQueue(); }
        });
      });
    }

    // Save the frames spot edits settled (nothing pending left): dry run, ask, save.
    var repeatsBtn = $('rv-accept-repeats');
    if (repeatsBtn) {
      repeatsBtn.addEventListener('click', function () {
        bulkAction({
          button: repeatsBtn, box: $('rv-accept-repeats-msg') || msg, url: '/v1/review/accept-repeats', keys: ALL_KEYS,
          counting: 'Counting the frames...',
          count: function (j) { return j.eligible || 0; },
          none: function () { return ['No repeat-only frame matches the filters (a frame with a box still to decide never is).']; },
          confirm: function (j, body, n) {
            return 'Save ' + n + ' repeat-only frame(s) as reviewed? Their boxes were all settled by "this spot on all frames"; ' +
              'nothing on them is pending.';
          },
          cancelled: 'Nothing saved.',
          done: function (r) { return r.message || ('Saved ' + (r.accepted || 0) + ' frame(s).'); },
          after: function () { return reloadQueue(); }
        });
      });
    }

    var rejectBtn = $('rv-reject-low');
    if (rejectBtn) {
      rejectBtn.addEventListener('click', function () {
        var box = $('rv-reject-low-msg');
        var score = parseFloat($('rv-reject-low-score').value);
        box.className = 'review-report';
        if (!(score >= 0.01 && score <= 1)) {
          box.className = 'review-report error-text';
          report(box, ['Enter a score between 0.01 and 1.']);
          return;
        }
        bulkAction({
          button: rejectBtn, box: box, url: '/v1/review/reject-low', keys: ALL_KEYS,
          start: { min_score: score },
          counting: 'Counting the boxes...',
          count: function (j) { return j.boxes || 0; },
          none: function (j) { return [j.message || ('No pending auto box below ' + score + ' on the unreviewed frames the filters select.')]; },
          confirm: function (j, body, n) {
            return 'Reject ' + n + ' box(es) on ' + (j.eligible || 0) + ' frame(s): the pending auto boxes ' +
              'scoring below ' + score + '? They are kept as hard negatives (reason low_score); the frames stay ' +
              'unreviewed and a reviewer can still change them per frame.';
          },
          cancelled: 'Nothing rejected.',
          done: function (r) { return r.message || ('Rejected ' + (r.boxes || 0) + ' box(es).'); },
          after: function () { return reloadQueue(); }
        });
      });
    }

    // `r` = reviewed_with_unwanted {frames, boxes, classes}: asks whether to reject the accepted
    // boxes of unwanted classes on reviewed frames too (POST /v1/review/reject-unwanted-reviewed)
    // and appends the outcome to `box`. Resolves either way.
    function askReviewedUnwanted(r, box) {
      if (!r || !(r.frames > 0)) return Promise.resolve();
      var text = r.frames + ' reviewed frame(s) have ' + r.boxes + ' accepted box(es) of ' +
        (r.classes || []).join(', ') + '. Reject those too? They are kept as evidence (reason other_class) ' +
        'and the frames stay reviewed; Cancel keeps them as training labels.';
      if (!window.confirm(text)) {
        box.appendChild(el('div', null, 'Reviewed frames left as they are (' + r.frames + ' with boxes of ' +
          (r.classes || []).join(', ') + ').'));
        return Promise.resolve();
      }
      return postJson('/v1/review/reject-unwanted-reviewed', {}).then(function (j) {
        box.appendChild(el('div', null, j.message || ('Rejected ' + (j.boxes || 0) + ' box(es).')));
        if (j.failed && j.failed.length) box.appendChild(el('div', 'error-text', j.failed.join('; ')));
        return reloadQueue();
      }).catch(function (e) {
        box.appendChild(el('div', 'error-text', e.message));
      });
    }

    var otherBtn = $('rv-reject-other');
    if (otherBtn) {
      otherBtn.addEventListener('click', function () {
        var box = $('rv-reject-other-msg');
        bulkAction({
          button: otherBtn, box: box, url: '/v1/review/reject-other-class', keys: ALL_KEYS,
          counting: 'Counting the boxes...',
          count: function (j) { return j.boxes || 0; },
          none: function (j) {
            return [j.message || 'No pending box of an unwanted class (hidden, or not in the class list) on the unreviewed frames the filters select.'];
          },
          afterNone: function (j) { return askReviewedUnwanted(j.reviewed_with_unwanted, box); },
          confirm: function (j, body, n) {
            var renamed = j.relabelled || 0;
            var text = 'Reject ' + (n - renamed) + ' pending model / auto box(es) of unwanted classes (hidden, or not in ' +
              'the class list) on ' + (j.eligible || 0) + ' frame(s)? They are kept as evidence (reason other_class); ' +
              'the frames stay unreviewed and a reviewer can still change them per frame.';
            if (renamed) {
              text += ' ' + renamed + ' box(es) whose label is a class of the list under another spelling ' +
                '(e.g. pottedplant) are renamed to the list\'s spelling instead.';
            }
            return text;
          },
          cancelled: 'Nothing rejected.',
          done: function (r) { return r.message || ('Changed ' + (r.boxes || 0) + ' box(es).'); },
          after: function (r) {
            return reloadQueue().then(function () { return askReviewedUnwanted(r.reviewed_with_unwanted, box); });
          }
        });
      });
    }

    // The overlap check: count, confirm, queue, follow. It writes suggestions only; the result line
    // offers the queue of overlapping boxes.
    function followOverlaps(id) {
      var box = $('rv-suggest-msg');
      pollTask(id, box, $('rv-suggest-progress'), function (t) {
        if (t.state !== 'done') return;
        box.appendChild(button('Show the overlapping frames', function () {
          showOverlaps();
          filterReload();
        }));
      });
    }

    var suggestBtn = $('rv-suggest-overlaps');
    if (suggestBtn) {
      suggestBtn.addEventListener('click', function () {
        var box = $('rv-suggest-msg');
        // "Start anyway" shows beside a memory refusal of the dry run (PrismMemory, live.js).
        var anyway = $('rv-suggest-anyway');
        var anywayTicked = PrismMemory.ticked(anyway);
        bulkAction({
          button: suggestBtn, box: box, url: '/v1/review/suggest-overlaps', keys: NO_SOURCE_KEYS,
          extra: function (body) {
            var large = $('rv-suggest-large');
            if (large && large.checked) body.allow_large = true;
            var askAgain = $('rv-suggest-again');
            if (askAgain && askAgain.checked) body.again = true;
          },
          counting: 'Counting the pairs...',
          count: function (j) { return j.pairs; },
          none: function (j) { return [j.message || 'No open overlap without a suggestion on these frames.']; },
          gate: function (j, body) {
            PrismMemory.sync(anyway, j.memory);
            if (j.memory && !anywayTicked) {
              box.className = 'review-report error-text';
              report(box, [j.memory.refusal, 'To queue it anyway, tick Start anyway above, then press Suggest overlaps again.']);
              return true;
            }
            if (j.memory) body.ignore_memory_check = true;
            return false;
          },
          confirm: function (j, body) {
            return (j.memory ? 'Warning: ' + j.memory.refusal + '\n\n' : '') + 'Check ' + j.pairs + ' overlapping pair(s) on ' + j.frames + ' frame(s), reviewed frames included' +
              (body.again ? ', answered pairs asked again' : '') + ', ' +
              'with RF-DETR L (' + j.crops + ' crops, on the training queue, about ' + Math.max(1, Math.ceil(j.crops * 0.08 / 60)) + ' min)? ' +
              'It writes suggestions only: no box changes until you confirm one here.';
          },
          cancelled: 'Nothing queued.',
          finish: function (r) {
            report(box, [r.message || 'Queued.']);
            if (r.id) followOverlaps(r.id);
          }
        });
      });
    }

    // Stacked boxes span reviewed frames too; the Frames filter does not narrow it (a reviewed frame
    // need not be in the review queue). Overlaps that are no stack stay a reviewer's: the server's
    // message says how many and how to list them.
    var stackedBtn = $('rv-reject-stacked');
    if (stackedBtn) {
      stackedBtn.addEventListener('click', function () {
        bulkAction({
          button: stackedBtn, box: $('rv-reject-stacked-msg'), url: '/v1/review/reject-stacked', keys: NO_SOURCE_KEYS,
          counting: 'Counting the boxes...',
          count: function (j) { return j.boxes || 0; },
          none: function (j) { return [j.message || 'No stacked duplicates on the frames these filters select.']; },
          confirm: function (j, body, n) {
            return 'Reject ' + n + ' stacked duplicate box(es) on ' + (j.eligible || 0) + ' frame(s), reviewed frames ' +
              'included? The best box of each stack stays. Frames stay reviewed and the boxes are kept as rejected evidence ' +
              '(reason duplicate, left out of the hard negatives); drawn boxes are never rejected.';
          },
          cancelled: 'Nothing rejected.',
          done: function (r) { return r.message || ('Rejected ' + (r.boxes || 0) + ' box(es).'); },
          after: function () {
            c.forgetHistories();   // the boxes rejected were decisions of these cameras
            return reloadQueue();
          }
        });
      });
    }

    // The undo of a re-label that reopened reviewed frames: back to reviewed when nothing is pending.
    var restoreBtn = $('rv-restore-reopened');
    if (restoreBtn) {
      restoreBtn.addEventListener('click', function () {
        bulkAction({
          button: restoreBtn, box: $('rv-restore-reopened-msg'), url: '/v1/review/restore-reopened', keys: ALL_KEYS,
          counting: 'Counting the frames...',
          count: function (j) { return j.eligible || 0; },
          none: function () { return ['No reopened frame with nothing left pending among the frames the filters select.']; },
          confirm: function (j, body, n) {
            return 'Put ' + n + ' reopened frame(s) back to reviewed? Their boxes stay as they are, and they ' +
              'return to exports. Frames with a pending box are not touched.';
          },
          cancelled: 'Nothing restored.',
          done: function (r) { return r.message || ('Restored ' + (r.restored || 0) + ' frame(s).'); },
          after: function () { return reloadQueue(); }
        });
      });
    }

    return { bulkAction: bulkAction, pollTask: pollTask };
  }

  globalThis.ReviewBulk = { describeTask: describeTask, init: init };
})();
