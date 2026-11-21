// Review tab, why a frame is next (/train/review; loaded before review.js): the sentences the
// details show for the Recommended order's `need` (GET /v1/review/next: the item's `need`,
// src/collect/labels/need.rs `RecallNeed`, and the answer's `need`, `NeedCounts`). Pure: no page,
// no browser API; review.js passes the item in. `tests/it/js_review_need.rs` runs this file in
// Node.
(function () {
  'use strict';

  // A count of boxes (a box at a parked spot counts a quarter): whole numbers as they are, else
  // up to two decimals.
  function num(n) {
    n = n || 0;
    return Number.isInteger(n) ? String(n) : String(Math.round(n * 100) / 100);
  }

  function boxes(n, what) {
    return num(n) + ' ' + what + (n === 1 ? '' : 'es');
  }

  function sceneName(s) { return s === 'ir' ? 'IR' : (s || ''); }

  // Why the Recommended order put `item` (its `need`, `camera`, `scene`, `event_rank`) where it
  // is, e.g. "raccoon · back · IR: 3 boxes on this camera (3 in all, floor 20), recall 22% on 9
  // held-out boxes (lowest on this camera), camera model missed it, new place in view, frame 3 of
  // a raccoon visit". '' without a need.
  function note(item) {
    var n = item && item.need;
    if (!n) return '';
    var where = item.camera + ' · ' + sceneName(item.scene);
    if (n.class == null) {
      var empty = 'nothing of a wanted class · ' + where + ': ' +
        num(n.empty_reviewed) + ' empty frame' + (n.empty_reviewed === 1 ? '' : 's') + ' reviewed in this scene';
      if (n.animal_share) {
        empty += ', ' + Math.round(n.animal_share * 100) + '% of this camera\'s reviewed frames here had an animal';
      }
      return empty;
    }
    var parts = [];
    parts.push(boxes(n.camera_reviewed, 'box') + ' on this camera' + (n.below_floor
      ? ' (' + num(n.class_reviewed) + ' in all, floor ' + n.floor + ')'
      : ' (' + num(n.reviewed) + ' on every camera in this scene)'));
    if (n.model_lacks) {
      parts.push('the trained model has no ' + n.class);
    } else if (n.recall != null) {
      var on = n.heldout_gt ? ' on ' + n.heldout_gt + ' held-out box' + (n.heldout_gt === 1 ? '' : 'es')
        : ' (estimated: no held-out box on this camera)';
      parts.push('recall ' + Math.round(n.recall * 100) + '%' + on + (n.catch_up ? ' (lowest on this camera)' : ''));
    } else if (n.catch_up) {
      parts.push('lowest recall on this camera');
    }
    if (n.missed) parts.push('camera model missed it');
    if (n.gap) parts.push('lost for a frame of its visit');
    if (n.spot) parts.push('at a parked spot (counts a quarter)');
    if (n.new_region) parts.push('new place in view');
    if (n.new_day) parts.push('new day');
    if (!n.strong && n.evidence != null && n.evidence < 0.5) parts.push('weak evidence (' + n.evidence.toFixed(2) + ')');
    if (n.proposed != null) parts.push((n.kept || 0) + ' of ' + n.proposed + ' proposals kept on this camera');
    if (n.expanded && item.event_rank) parts.push('frame ' + (item.event_rank + 1) + ' of a ' + n.class + ' visit');
    return n.class + ' · ' + where + ': ' + parts.join(', ');
  }

  // The answer's `need` (reviewed boxes per class and scene, empty frames per scene) in one line:
  // "person 12 (day 8, night 4) · raccoon 3 (IR 3) · empty: day 20".
  function counts(c) {
    if (!c) return '';
    var per = c.boxes || {};
    var list = Object.keys(per).sort().map(function (k) {
      var s = per[k] || {};
      var keys = Object.keys(s).sort();
      var total = keys.reduce(function (a, x) { return a + (s[x] || 0); }, 0);
      return k + ' ' + num(total) + ' (' + keys.map(function (x) { return sceneName(x) + ' ' + num(s[x]); }).join(', ') + ')';
    });
    var e = c.empty || {};
    var ek = Object.keys(e).sort();
    if (ek.length) list.push('empty: ' + ek.map(function (x) { return sceneName(x) + ' ' + e[x]; }).join(', '));
    return list.length ? list.join(' · ') : 'none yet';
  }

  globalThis.ReviewNeed = { note: note, counts: counts };
})();
