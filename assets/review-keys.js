// Review tab, the keys (/train/review; loaded before review.js): one list, `KEYS`, for the table
// under Panel options and the `?` overlay (`#rv-keys-dialog`), both drawn by `init`, which also
// wires the overlay (the Keys ? button opens it, a click on its backdrop closes it, closing gives
// the frame the focus back) and returns `open`. `typing` is whether a field takes a key itself
// (review.js's keydown leaves those keys alone). The class row's span (`.rv-key-classes`) is
// filled by review.js's fillKeyHint. tests/it/js_review_keys.rs runs this file in Node.
(function () {
  'use strict';

  // [group, key, text]; a text of null is the class row (filled by fillKeyHint), a key of '' a note.
  var KEYS = [
    ['Decide', 'a', 'accept the selected box (keep a drawn one) and go to the next'],
    ['Decide', 'r', 'reject the selected box (delete a drawn one) and go to the next'],
    ['Decide', 't', 'real, parked: accept the selected box and remember its spot, so that auto-label runs accept it there (a VLM no stays flagged); not on drawn boxes'],
    ['Decide', '1-9, 0', null],
    ['Decide', 'u', 'reject the selected box as a duplicate of the box it overlaps (stacked, nested or partly overlapping)'],
    ['Decide', 'Shift+U', 'stacked boxes: keep the best of each stack, reject the others as duplicates'],
    ['Decide', 'y', 'overlap suggestion: accept the one on screen (drops the box marked drop?, as a duplicate; or keeps both as two objects). Doing anything else ignores it'],
    ['Decide', 'y (dataset check)', 'once no overlap suggestion is on screen, accept the dataset check\'s suggestion (strongest first): add the dashed box, take the other class or the tighter box, or keep a label no model found. Doing anything else leaves it open'],
    ['Decide', 'w', 'dataset check suggestion: no (not an object, keep the label or the box as they are, or dismiss "no model found"); remembered, the next check does not ask again'],
    ['Decide', 'd', 'unsure: a real object you cannot name for sure: kept out of training, its frame goes to held-out only, ignored by the benchmark (?! on the box); again: not unsure'],
    ['Decide', 'Shift+1-9', 'after a rejection, or on a rejected box: why it was wrong (optional; nothing there, insect / web, rain / snow, IR glare, reflection, part of the scene, other object, too small, duplicate). Too small is also a button in the decision card: a real object too small to label, no negative, treated like unsure'],
    ['Decide', 'c', 'accept likely: every likely (★) box on the frame; the others stay pending'],
    ['Decide', 'Shift+A', 'accept proposals: the pending boxes saving would accept (flagged ? boxes and hints stay pending)'],
    ['Decide', 'Shift+R', 'reject the rest: every box still undecided, the way to clear the unlikely boxes'],
    ['Decide', 'e', 'confirmed empty (rejects every box)'],
    ['Decide', 'Delete', 'delete a drawn box, reject a model box'],
    ['Decide', 'z', 'undo'],
    ['Decide', 's, Enter', 'save & next; flagged (?) boxes need a decision first, then it asks about boxes not checked yet and "Anything missing?" (Panel options)'],
    ['Decide', 'drag', 'on an edge or corner of the selected box (it lights up under the pointer): resize; inside it: move; elsewhere: draw a box'],
    ['Decide', 'arrows', 'move the selected box 1 px (Shift: 10 px)'],
    ['Decide', '', 'The next box is the next likely (★) one, then the next flagged (?) one, then the next pending one, then each accepted / rejected box not checked yet (check pass, marked to check).'],
    ['Navigate', 'j / k', 'select the next / previous box'],
    ['Navigate', 'Esc', 'no box selected (during the quarter look: the whole frame first)'],
    ['Navigate', 'n', 'next frame without saving (after Back: the frame you came from)'],
    ['Navigate', 'p', 'back to the previous frame of this session (as saved)'],
    ['Navigate', 'x', 'zoom to the selected box and follow the selection (the next box, the next frame) until f'],
    ['Navigate', 'f', 'fit: the whole frame, for the next boxes and frames too'],
    ['Navigate', 'q', 'while "Anything missing?" asks: zoom to the top-left, top-right, bottom-left and bottom-right quarter in turn, then the whole frame again (small, distant objects are the ones most often missed; drawing works there, s saves)'],
    ['Navigate', 'wheel', 'zoom about the pointer (up to 16x; sharp pixels from 3x on screen); after zooming or panning by hand the next box is zoomed to, and that stays the choice until f'],
    ['Navigate', 'right / middle drag, Space + drag', 'pan'],
    ['Navigate', 'overview', 'drag the outlined part to pan, the wheel zooms there; click a box to select it, elsewhere to move the view there'],
    ['View', 'b', 'image (view only): auto, off, bright, bright + equalised; auto = bright + equalised on night / IR frames'],
    ['View', '[ / ]', 'image brightness down / up (view only; the View menu has the slider): lifts or deepens the shadows, never clips the highlights'],
    ['View', '{ / }', 'image contrast down / up (Shift+[ / ]; view only; the View menu has the slider), about the frame\'s median level (a dark frame does not go black)'],
    ['View', 'Shift+0', 'image back to auto, brightness and contrast 0'],
    ['View', 'v', 'spotlight on / off: the frame dimmed outside the selected box, the other boxes faint'],
    ['View', 'o', 'the overview of the whole frame: at the top of the side panel, or in a corner of the frame (wide screens; kept for your next visits)'],
    ['View', 'h (hold)', 'hide the boxes'],
    ['View', 'l', 'labels on the boxes on / off'],
    ['View', 'i', 'show / put away the hidden boxes of this frame (low-score auto boxes and boxes at spots you rejected, dimmed; accepting one makes it a label)'],
    ['View', '?', 'this list (Esc closes it)']
  ];

  // A field that takes the key itself. A checkbox / radio does not (the "Confirmed empty" box
  // keeps focus after a click); a box's class select takes only its own navigation keys.
  function typing(t, k) {
    var tag = t && t.tagName;
    if (tag === 'INPUT') return !/^(checkbox|radio|button|submit|reset)$/.test(t.type);
    if (tag === 'SELECT' && t.classList.contains('bx-label')) {
      return /^(Arrow(Up|Down|Left|Right)|Enter| |Home|End|PageUp|PageDown|Tab)$/.test(k);
    }
    return tag === 'TEXTAREA' || tag === 'SELECT' || !!(t && t.isContentEditable);
  }

  // Draws the key table and the overlay and wires the overlay. `c`: `el`, `button` (dom.js /
  // review.js), `dialog` (the overlay), `canvas` (the focus after it closes) and `keysBtn` (the
  // Keys ? button, may be null). Returns `open` (the `?` key).
  function init(c) {
    var el = c.el, button = c.button, keysDialog = c.dialog, canvas = c.canvas, keysBtn = c.keysBtn;

    function keyRows(tbody, group, withId) {
      KEYS.forEach(function (k) {
        if (group && k[0] !== group) return;
        var tr = el('tr');
        tr.appendChild(el('td', null, k[1]));
        var td = el('td');
        if (k[2] == null) {
          td.appendChild(document.createTextNode('class 1-10'));
          var span = el('span', 'rv-key-classes');
          if (withId) span.id = 'rv-key-classes';
          td.appendChild(span);
          td.appendChild(document.createTextNode(': relabel the selected box (accepting it) and go to the next, else the class of new boxes; the bar\'s It\'s a… lists them all'));
        } else td.textContent = k[2];
        if (!k[1]) td.className = 'muted';
        tr.appendChild(td);
        tbody.appendChild(tr);
      });
    }

    function renderKeys() {
      var tbody = document.querySelector('.review-keys tbody');
      if (tbody) {
        tbody.textContent = '';
        ['Decide', 'Navigate', 'View'].forEach(function (g) {
          var tr = el('tr', 'rv-keys-group');
          var th = el('th', null, g);
          th.colSpan = 2;
          tr.appendChild(th);
          tbody.appendChild(tr);
          keyRows(tbody, g, true);
        });
      }
      keysDialog.textContent = '';
      // Close first: the dialog focuses it on open, and it should not scroll the list to its end.
      var head = el('div', 'rv-keys-head');
      head.appendChild(el('h3', null, 'Keyboard'));
      head.appendChild(button('Close', function () { keysDialog.close(); }, 'secondary'));
      keysDialog.appendChild(head);
      var cols = el('div', 'rv-keys-cols');
      ['Decide', 'Navigate', 'View'].forEach(function (g) {
        var col = el('section');
        col.appendChild(el('h4', null, g));
        var t = el('table', 'review-keys');
        var tb = el('tbody');
        keyRows(tb, g, false);
        t.appendChild(tb);
        col.appendChild(t);
        cols.appendChild(col);
      });
      keysDialog.appendChild(cols);
    }

    function openKeys() {
      if (keysDialog.open) return;
      if (keysDialog.showModal) keysDialog.showModal(); else keysDialog.setAttribute('open', '');
    }
    keysDialog.addEventListener('click', function (ev) { if (ev.target === keysDialog) keysDialog.close(); });
    keysDialog.addEventListener('close', function () { canvas.focus({ preventScroll: true }); });
    if (keysBtn) keysBtn.addEventListener('click', openKeys);
    renderKeys();
    return { open: openKeys };
  }

  globalThis.ReviewKeys = { KEYS: KEYS, typing: typing, init: init };
})();
