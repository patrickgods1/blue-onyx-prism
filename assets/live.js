// Blue Onyx Prism: live config for every page. Polls GET /v1/config (ETag / If-None-Match, every
// 2 s while the tab is visible and on focus; no timer runs while it is hidden; after a failure
// the header says so and the period backs off to 30 s), keeps the header's "pending changes - restart to
// apply" indicator current, and hands each new payload to the page scripts
// (PrismLive.onConfig). Also the form helpers the pages use to refresh fields the user has not
// edited while keeping (and marking) the ones they have.
// PrismPoll: the one scheduler for the pages' poll loops (Logs, Stats, Home, Train, Config). A chained
// timeout, never an interval: the next run is scheduled when the last one has finished, so a slow
// server never sees two requests of one loop at once. No timer runs while the tab is hidden; the
// loop runs at once when it becomes visible again; errors back off (period x 2, 4, 8, ... up to
// `max`); `kick()` asks for a run now (one more right after the current one if it is in flight).
// PrismPoll.create({run, period, max, more, env}): `run()` returns a promise (a rejection counts
// as a failure); `more(value)` false parks the loop until `kick()` (polling only while busy).
// `env` is for tests: {hidden(), set(fn, ms), clear(id), onVisible(cb)}.
// Signed-in requests: every same-origin `fetch` of the pages goes through the wrapper below. A
// request made by a poll loop (synchronously inside a PrismPoll run, inside
// PrismPoll.background(fn), or through PrismPoll.fetch) carries `X-Prism-Poll: 1`, so it does not
// keep the session alive (an unattended tab signs out after the idle hour), and gives up after
// POLL_TIMEOUT_MS (a request left hanging by a network switch would stall its loop; the abort is
// an error, so the loop backs off). A fetch chained after a poll's answer (inside a `.then`) is no
// longer synchronous: poll code uses PrismPoll.fetch or PrismPoll.background for it. A page load
// cannot carry a header, so a reload or navigation a poll makes (not the user) goes through
// PrismPoll.navigate, which sets the short-lived `__Host-bop_poll` cookie the server reads as the
// poll mark (and clears in its answer). A 503 ("sign-in is closed") stays an error for the page
// to show.
// A 401 (the session ended) does not leave the page: one shared sign-in dialog opens (only while
// the page is visible) and the refused requests wait behind it. Signed in (POST /login answers
// JSON with the new cookies), each is sent once more and its caller gets that answer, as if the
// first had passed (the 401 came before the server did anything). While the dialog is open, poll
// requests are held, not sent, and go out after it. The way to /login?next=<this page, with its
// #hash> stays for what the dialog cannot do: Cancel (or Esc), a request whose body cannot be
// sent twice (only none, a string, FormData, URLSearchParams or a Blob can), a 401 again right
// after signing in, a browser without <dialog>, and the /login and /setup pages themselves (they
// never send to themselves). Requests left waiting by that redirect get their 401 if the page is
// still there RELEASE_MS later (a leave prompt kept it), and the next 401 opens the dialog again.
// PrismPoll.signIn() opens the dialog for a caller that has no request to repeat (the config poll's
// signed-out answer, a native form; live.js below) and resolves once signed in.
// PrismDraft: unsaved edits kept in sessionStorage across a tab the browser discarded (phones do,
// and `beforeunload` does not run there): save / load / drop by key (prefixed with this path),
// every access in try/catch (no storage, a full one or a refusing one is nothing kept), and
// `verdict(draft, id, rev)`: "none" (no draft), "other" (another item's), "restore" (the same
// item at the same revision) or "drop" (it changed since).
(function (root) {
  "use strict";
  var polling = 0, pollFetch = null, realFetch = null;
  var POLL_TIMEOUT_MS = 15000, RELEASE_MS = 5000, REDIRECT_ONCE_MS = 2000;
  function sameOrigin(url) {
    if (typeof url !== "string") return false;
    if (url.charAt(0) === "/") return url.charAt(1) !== "/" && url.charAt(1) !== "\\";
    var o = root.location && root.location.origin;
    return !!o && url.indexOf(o + "/") === 0;
  }
  function copyOf(init) {
    var out = {}, k;
    for (k in (init || {})) if (Object.prototype.hasOwnProperty.call(init, k)) out[k] = init[k];
    return out;
  }
  function withPollHeader(init) {
    init = init || {};
    var out = copyOf(init), k;
    var h = init.headers;
    if (typeof Headers !== "undefined" && h instanceof Headers) {
      h = new Headers(h); h.set("X-Prism-Poll", "1");
    } else {
      var copy = {};
      for (k in (h || {})) if (Object.prototype.hasOwnProperty.call(h, k)) copy[k] = h[k];
      copy["X-Prism-Poll"] = "1"; h = copy;
    }
    out.headers = h;
    return out;
  }
  // The real fetch; a poll's with its own timeout (a fresh one per send, so a repeat after the
  // sign-in dialog is not cut short by the first send's clock). A caller's own signal wins.
  function send(input, init, timed) {
    if (!timed || typeof root.AbortController !== "function" || (init && init.signal)) return realFetch.call(root, input, init);
    var c = new root.AbortController(), opts = copyOf(init);
    opts.signal = c.signal;
    var t = root.setTimeout(function () { c.abort(); }, POLL_TIMEOUT_MS);
    var stop = function () { root.clearTimeout(t); };
    return realFetch.call(root, input, opts).then(function (r) { stop(); return r; }, function (e) { stop(); throw e; });
  }

  // ---- the sign-in dialog -----------------------------------------------------------------
  // queue: requests waiting for it ({input, init, poll, response (null: held, never sent),
  // resolve, reject}); waiters: signIn() callers; parked: requests left by a redirect.
  var gate = {open: false, ui: null, queue: [], waiters: [], parked: [], redirectedAt: -Infinity, release: null, watching: false};
  function doc() { return root.document; }
  function authPage() { var l = root.location; return !l || l.pathname === "/login" || l.pathname === "/setup"; }
  function pageHidden() { var d = doc(); return !!(d && d.hidden); }
  function redirect() {
    var l = root.location;
    if (!l || authPage()) return;
    var now = Date.now();
    if (now - gate.redirectedAt > REDIRECT_ONCE_MS) {
      gate.redirectedAt = now;
      l.assign("/login?next=" + encodeURIComponent(l.pathname + (l.search || "") + (l.hash || "")));
    }
    if (gate.release === null && typeof root.setTimeout === "function") gate.release = root.setTimeout(release, RELEASE_MS);
  }
  // Still on the page after a redirect (a leave prompt kept it): what waited gets its 401 (a held
  // poll goes out again, through the wrapper), so the page can say so and try again.
  function release() {
    gate.release = null;
    gate.parked.splice(0).forEach(function (e) {
      if (e.response) e.resolve(e.response);
      else e.resolve((e.poll ? pollFetch : root.fetch)(e.input, e.init));
    });
  }
  // Give up on the dialog: everything waiting goes with the page.
  function leave() {
    gate.parked = gate.parked.concat(gate.queue.splice(0));
    gate.waiters = [];
    redirect();
  }
  // A body that can be sent a second time.
  function replayable(input, init) {
    if (typeof input !== "string") return false;
    var b = init ? init.body : null;
    return b === undefined || b === null || typeof b === "string" ||
      (typeof root.FormData === "function" && b instanceof root.FormData) ||
      (typeof root.URLSearchParams === "function" && b instanceof root.URLSearchParams) ||
      (typeof root.Blob === "function" && b instanceof root.Blob);
  }
  function canDialog() {
    var d = doc();
    return !!(d && typeof d.createElement === "function" && typeof root.HTMLDialogElement === "function" && !authPage());
  }
  function say(ui, text, quiet) {
    ui.msg.textContent = text || "";
    ui.msg.hidden = !text;
    ui.msg.className = "signin-msg " + (quiet ? "muted" : "error-text");
  }
  function focus(el) { try { el.focus({preventScroll: true}); } catch (e) { try { el.focus(); } catch (e2) { /* not focusable */ } } }
  // Built once, on the first 401, with the page's own form styles (style.css `dialog.signin-dialog`).
  // A real <form> with username + current-password fields, so password managers fill it.
  function build() {
    var d = doc(), dlg = d.createElement("dialog");
    if (typeof dlg.showModal !== "function") return null;
    var make = function (tag, cls, text) {
      var e = d.createElement(tag);
      if (cls) e.className = cls;
      if (text !== undefined) e.textContent = text;
      return e;
    };
    var attrs = function (e, map) { Object.keys(map).forEach(function (k) { e.setAttribute(k, map[k]); }); return e; };
    dlg.className = "signin-dialog";
    attrs(dlg, {"aria-labelledby": "signin-title", "aria-describedby": "signin-why"});
    var form = attrs(make("form", "signin-form"), {method: "post", action: "/login", "data-no-auth-check": ""});
    var title = make("h2", null, "Sign in again"); title.id = "signin-title";
    var why = make("p", "muted", "Your session ended (signed out, or idle too long). Sign in to carry on: this page and what you changed on it stay as they are.");
    why.id = "signin-why";
    var user = attrs(make("input"), {type: "text", name: "username", autocomplete: "username", autocapitalize: "none", autocorrect: "off", spellcheck: "false", maxlength: "32", required: ""});
    var pass = attrs(make("input"), {type: "password", name: "password", autocomplete: "current-password", maxlength: "512", required: ""});
    var userLabel = make("label", null, "Username "), passLabel = make("label", null, "Password ");
    userLabel.appendChild(user); passLabel.appendChild(pass);
    var msg = attrs(make("p", "signin-msg"), {role: "alert"});
    msg.hidden = true;
    var actions = make("div", "actions");
    var submit = make("button", null, "Sign in"); submit.type = "submit";
    var cancel = make("button", "secondary", "Cancel"); cancel.type = "button";
    actions.appendChild(submit); actions.appendChild(cancel);
    var note = make("p", "muted signin-note", "Cancel opens the sign-in page instead; changes this page has not saved are lost then.");
    [title, why, userLabel, passLabel, msg, actions, note].forEach(function (e) { form.appendChild(e); });
    dlg.appendChild(form);
    var ui = {dlg: dlg, form: form, user: user, pass: pass, msg: msg, submit: submit, cancel: cancel, busy: false};
    form.addEventListener("submit", function (ev) { ev.preventDefault(); attempt(ui); });
    cancel.addEventListener("click", function () { cancelDialog(); });
    // Esc is Cancel (the dialog never closes on its own with requests still waiting).
    dlg.addEventListener("cancel", function (ev) { ev.preventDefault(); cancelDialog(); });
    return ui;
  }
  function show() {
    if (gate.open) return;
    if (!canDialog()) { leave(); return; }
    if (pageHidden()) { whenVisible(); return; }
    var ui = gate.ui || (gate.ui = build());
    if (!ui) { leave(); return; }
    var d = doc();
    if (!ui.dlg.parentNode && d.body) d.body.appendChild(ui.dlg);
    var shown = d.querySelector && d.querySelector(".signed-in");
    var name = shown ? String(shown.textContent || "").trim() : "";
    if (name && !ui.user.value) ui.user.value = name;
    say(ui, "");
    try { ui.dlg.showModal(); } catch (e) { leave(); return; }
    gate.open = true;
    focus(ui.user.value ? ui.pass : ui.user);
  }
  // The dialog opens once the page is visible again (a hidden tab's request waits).
  function whenVisible() {
    var d = doc();
    if (gate.watching || !d || typeof d.addEventListener !== "function") return;
    gate.watching = true;
    var on = function () {
      if (pageHidden()) return;
      d.removeEventListener("visibilitychange", on);
      gate.watching = false;
      if (gate.queue.length || gate.waiters.length) show();
    };
    d.addEventListener("visibilitychange", on);
  }
  function closeDialog() {
    gate.open = false;
    var ui = gate.ui;
    if (!ui) return;
    ui.pass.value = "";
    say(ui, "");
    try { ui.dlg.close(); } catch (e) { /* already closed */ }
  }
  function cancelDialog() { closeDialog(); leave(); }
  function attempt(ui) {
    if (ui.busy) return;
    var name = String(ui.user.value || "").trim(), pw = ui.pass.value || "";
    if (!name || !pw) { say(ui, "Type your username and password."); return; }
    ui.busy = true; ui.submit.disabled = true;
    say(ui, "Signing in…", true);
    realFetch.call(root, "/login", {
      method: "POST",
      credentials: "same-origin",
      headers: {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
      body: new root.URLSearchParams({username: name, password: pw, next: ""})
    }).then(function (r) {
      return r.json().then(null, function () { return null; }).then(function (j) {
        if (j && j.success) { signedIn(); return; }
        say(ui, (j && (j.error || j.message)) || ("Not signed in: the server answered HTTP " + r.status + "."));
        if (r.status === 401) { ui.pass.value = ""; focus(ui.pass); }
      });
    }, function () {
      say(ui, "Not signed in: the server did not answer. Check that it runs, then try again.");
    }).then(function () { ui.busy = false; ui.submit.disabled = false; });
  }
  // Signed in: every waiting request goes out once more; a 401 again leaves for /login.
  function signedIn() {
    closeDialog();
    var q = gate.queue.splice(0), w = gate.waiters.splice(0);
    q.forEach(function (e) {
      send(e.input, e.init, e.poll).then(function (r) {
        if (r && r.status === 401) { e.response = r; gate.parked.push(e); redirect(); return; }
        e.resolve(r);
      }, e.reject);
    });
    w.forEach(function (resolve) { resolve(); });
  }
  // A same-origin 401: wait behind the dialog (or leave for /login, see above).
  function refused(input, init, response, poll) {
    return new Promise(function (resolve, reject) {
      if (authPage()) return;   // never settles: nothing loops back to the login page
      var e = {input: input, init: init, poll: poll, response: response, resolve: resolve, reject: reject};
      if (!replayable(input, init) || !canDialog()) { gate.parked.push(e); redirect(); return; }
      gate.queue.push(e);
      show();
    });
  }
  function signIn() {
    return new Promise(function (resolve) {
      if (authPage()) return;
      if (!canDialog()) { redirect(); return; }
      gate.waiters.push(resolve);
      show();
    });
  }
  function wrap(real, poll) {
    return function (input, init) {
      var url = typeof input === "string" ? input : (input && input.url) || "";
      var mine = sameOrigin(url);
      var marked = mine && (poll || polling > 0) && typeof input === "string";
      if (marked) init = withPollHeader(init);
      // Polls wait while the dialog is open: they go out after it.
      if (marked && gate.open) {
        return new Promise(function (resolve, reject) {
          gate.queue.push({input: input, init: init, poll: true, response: null, resolve: resolve, reject: reject});
        });
      }
      return send(input, init, marked).then(function (r) {
        if (mine && r && r.status === 401) return refused(input, init, r, marked);
        return r;
      });
    };
  }
  if (typeof root.fetch === "function" && !root.fetch.prismWrapped) {
    realFetch = root.fetch;
    root.fetch = wrap(realFetch, false);
    root.fetch.prismWrapped = true;
    pollFetch = wrap(realFetch, true);
  }
  // Run `fn` as a poll: the requests it starts synchronously carry the poll header.
  function background(fn) {
    polling++;
    try { return fn(); } finally { polling--; }
  }
  // A reload (`url` empty) or navigation a poll makes: marked as no activity of the user.
  function navigate(url, replace) {
    var loc = root.location;
    if (!loc) return;
    try {
      if (typeof document !== "undefined") document.cookie = "__Host-bop_poll=1; Secure; SameSite=Strict; Path=/; Max-Age=30";
    } catch (e) { /* no cookies: the load counts as activity */ }
    if (!url) loc.reload(); else if (replace) loc.replace(url); else loc.assign(url);
  }
  function delay(period, failures, max) {
    return failures ? Math.min(max, period * Math.pow(2, Math.min(failures, 5))) : period;
  }
  var browser = {
    hidden: function () { return typeof document !== "undefined" && document.hidden; },
    set: function (f, ms) { return setTimeout(f, ms); },
    clear: function (id) { clearTimeout(id); },
    onVisible: function (cb) {
      if (typeof document !== "undefined") document.addEventListener("visibilitychange", cb);
    },
    // Back from the back-forward cache, or the network is back: worth asking at once.
    onWake: function (cb) {
      if (typeof root.addEventListener !== "function") return;
      root.addEventListener("pageshow", function (e) { if (e && e.persisted) cb(); });
      root.addEventListener("online", function () { cb(); });
    }
  };
  function create(o) {
    var env = o.env || browser, max = o.max || 30000;
    var timer = null, busy = false, again = false, failures = 0, stops = 0, parked = true; // parked until start()
    function clear() { if (timer !== null) { env.clear(timer); timer = null; } }
    function schedule() {
      clear();
      if (parked || env.hidden()) return;
      timer = env.set(function () { timer = null; go(); }, delay(o.period, failures, max));
    }
    function go() {
      clear();
      if (busy) { again = true; return; }
      if (env.hidden()) return;
      busy = true; again = false;
      var seen = stops; // a stop() while the run is out wins over its answer
      var p;
      try { p = Promise.resolve(background(o.run)); } catch (e) { p = Promise.reject(e); }
      p.then(function (v) {
        failures = 0;
        if (o.more && seen === stops) parked = !o.more(v);
      }, function () {
        failures++;
        if (o.onError) { try { o.onError(failures); } catch (e) { /* the loop goes on */ } }
      }).then(function () {
        busy = false;
        if (again && !parked && !env.hidden()) { again = false; go(); } else schedule();
      });
    }
    env.onVisible(function () {
      if (env.hidden()) clear(); else if (!parked) go();
    });
    // A run in flight answers (or times out) by itself: no second one beside it.
    if (env.onWake) env.onWake(function () { if (!parked && !busy && !env.hidden()) go(); });
    return {
      start: function () { parked = false; go(); },
      kick: function () { parked = false; go(); },
      stop: function () { stops++; parked = true; clear(); },
      state: function () { return {busy: busy, timer: timer !== null, parked: parked, failures: failures}; }
    };
  }
  root.PrismPoll = {
    create: create,
    delay: delay,
    background: background,
    inPoll: function () { return polling > 0; },
    navigate: navigate,
    signIn: signIn,
    // The dialog is open or requests wait for it (the Test page's upload clock waits too).
    signingIn: function () { return gate.open || gate.queue.length > 0; },
    TIMEOUT_MS: POLL_TIMEOUT_MS,
    fetch: function (url, init) { return pollFetch ? pollFetch(url, init) : root.fetch(url, withPollHeader(init)); }
  };

  // ---- drafts ------------------------------------------------------------------------------
  function storage() { try { return root.sessionStorage || null; } catch (e) { return null; } }
  function draftKey(key) { var l = root.location; return "prism:draft:" + ((l && l.pathname) || "") + ":" + key; }
  root.PrismDraft = {
    save: function (key, value) {
      try { var s = storage(); if (!s) return false; s.setItem(draftKey(key), JSON.stringify(value)); return true; } catch (e) { return false; }
    },
    load: function (key) {
      try {
        var s = storage(); if (!s) return null;
        var t = s.getItem(draftKey(key));
        return t === null || t === undefined ? null : JSON.parse(t);
      } catch (e) { return null; }
    },
    drop: function (key) {
      try { var s = storage(); if (s) s.removeItem(draftKey(key)); } catch (e) { /* nothing kept */ }
    },
    verdict: function (draft, id, rev) {
      if (!draft || typeof draft !== "object") return "none";
      if (draft.id !== id) return "other";
      return draft.rev === rev ? "restore" : "drop";
    }
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
(function () {
  "use strict";
  if (typeof document === "undefined") return; // loaded by the node test for PrismPoll only
  var PERIOD = 2000, MAX_PERIOD = 30000, RESTART_GIVE_UP_MS = 120000;
  // On a phone the header's links are one row that scrolls: the current page in view.
  var navRow = document.querySelector("header nav"), navHere = navRow && navRow.querySelector("a.active");
  if (navHere) navRow.scrollLeft = navHere.offsetLeft - navRow.offsetLeft - (navRow.clientWidth - navHere.offsetWidth) / 2;

  // ---- scroll cues ------------------------------------------------------------------------------
  // `[data-scroll-cue]`: a row that scrolls sideways (the phone header, the Train tabs, Review's
  // reason chips) fades the edge where more is hidden (`more-left` / `more-right` in style.css).
  function cueOne(el) {
    if (!el || !el.classList) return;
    var max = (el.scrollWidth || 0) - (el.clientWidth || 0), x = Math.abs(el.scrollLeft || 0);
    el.classList.toggle("more-left", max > 1 && x > 1);
    el.classList.toggle("more-right", max > 1 && x < max - 1);
  }
  function scrollCue(el) {
    if (!el || !el.addEventListener) return;
    if (!el._cue) {
      el._cue = true;
      el.addEventListener("scroll", function () { cueOne(el); }, {passive: true});
    }
    cueOne(el);
    // A row built by a script is measured once it is in the page.
    if (window.requestAnimationFrame) window.requestAnimationFrame(function () { cueOne(el); });
  }
  function scrollCues() {
    if (document.querySelectorAll) document.querySelectorAll("[data-scroll-cue]").forEach(scrollCue);
  }
  scrollCues();
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scrollCues);
  window.addEventListener("load", scrollCues);
  window.addEventListener("resize", scrollCues);
  var listeners = [], etag = null, last = null, busy = null, failures = 0, timer = null;
  // The session as the polls last saw it: `lastGood` = when a signed-in answer came (0: none
  // yet), `signedOut` = the answer turned minimal (the sign-in dialog is asked for).
  var lastGood = 0, signedOut = false;
  function fullAnswer(j) { return !!j && j.config !== undefined; }

  // Listeners run on a poll's answer: what they ask the server synchronously is a poll too.
  function emit(data, prev) {
    listeners.forEach(function (fn) {
      try { PrismPoll.background(function () { fn(data, prev); }); } catch (e) { if (window.console) console.error(e); }
    });
  }
  // Why a request failed, for the user: "HTTP 403", "not JSON (HTTP 200)", "no answer".
  function why(e) { return (e && e.reason) || "no answer"; }
  function failure(reason) { var e = new Error(reason); e.reason = reason; return e; }
  // JSON body of a response; rejects with a readable reason on an error status or another body.
  function json(r) {
    if (!r.ok) return Promise.reject(failure("HTTP " + r.status + (r.statusText ? " " + r.statusText : "")));
    return r.json().catch(function () { throw failure("not JSON (HTTP " + r.status + ")"); });
  }
  // Header note while the server does not answer usefully (empty text hides it).
  function liveStatus(text) {
    var s = document.getElementById("live-status");
    if (!s) return;
    s.hidden = !text; s.textContent = text || "";
  }
  // Fetch /v1/config; resolves to the payload (the last one on 304 or error).
  function poll(force) {
    if (busy) return busy;
    var h = {"Accept": "application/json"};
    if (etag && !force) h["If-None-Match"] = etag;
    busy = PrismPoll.fetch("/v1/config", {headers: h, cache: "no-store"}).then(function (r) {
      if (r.status === 304) { if (fullAnswer(last)) lastGood = Date.now(); return last; }
      return json(r).then(function (j) {
        // The full answer turned minimal: the session ended (idle, 12 h, signed out elsewhere).
        // The dialog opens (polls wait meanwhile); signed in, the next poll is the full answer.
        if (fullAnswer(last) && !fullAnswer(j)) {
          signedOut = true;
          PrismPoll.signIn().then(function () { signedOut = false; poll(true).then(schedule); });
          return last;
        }
        if (fullAnswer(j)) { lastGood = Date.now(); signedOut = false; }
        etag = r.headers.get("ETag"); var prev = last; last = j; emit(j, prev); return j;
      });
    }).then(function (j) {
      failures = 0; liveStatus(null); return j;
    }, function (e) {
      failures++;
      liveStatus("Live updates: server " + (e && e.reason ? "answered " + why(e) : "not answering") + "; retrying in " + Math.round(delay() / 1000) + " s");
      return last;
    }).then(function (j) { busy = null; return j; });
    return busy;
  }
  // 2 s; after failures 4, 8, 16, then 30 s.
  function delay() { return failures ? Math.min(MAX_PERIOD, PERIOD * Math.pow(2, Math.min(failures, 5))) : PERIOD; }
  // One timer at a time, none while the tab is hidden.
  function schedule() {
    clearTimeout(timer); timer = null;
    if (document.hidden) return;
    timer = setTimeout(function () { timer = null; poll().then(schedule); }, delay());
  }

  // Header: "N pending changes - restart to apply".
  function header(j) {
    var box = document.getElementById("pending-restart");
    if (!box) return;
    var n = j.pendingCount || 0, a = box.querySelector("a");
    box.hidden = n === 0;
    a.textContent = n + " pending change" + (n === 1 ? "" : "s") + " — restart to apply";
    a.title = ((j.pending && j.pending.all) || []).join("\n");
  }
  listeners.push(header);

  // ---- reloads that keep the scroll position --------------------------------------------------
  // A reload the page triggers itself (a restart finished, a model finished loading) comes
  // back where the user was: the position is kept in sessionStorage for this path and restored
  // once, after load (and again while asynchronously rendered content is still growing).
  var SCROLL_KEY = "prism:scroll:" + location.pathname;
  // `quiet`: a poll asked for it, not the user (PrismPoll.navigate).
  function reloadKeep(url, quiet) {
    try { sessionStorage.setItem(SCROLL_KEY, JSON.stringify({x: window.scrollX, y: window.scrollY, t: Date.now()})); } catch (e) { /* private mode */ }
    if (quiet) PrismPoll.navigate(url, true);
    else if (url) location.replace(url); else location.reload();
  }
  (function restoreScroll() {
    var saved = null;
    try { saved = JSON.parse(sessionStorage.getItem(SCROLL_KEY) || "null"); sessionStorage.removeItem(SCROLL_KEY); } catch (e) { saved = null; }
    if (!saved || Date.now() - saved.t > 10 * 60 * 1000) return;
    if ("scrollRestoration" in history) history.scrollRestoration = "manual";
    var tries = 0, user = false;
    ["wheel", "touchstart", "keydown", "mousedown"].forEach(function (t) { window.addEventListener(t, function () { user = true; }, {once: true, passive: true}); });
    function go() {
      if (user) return;
      window.scrollTo(saved.x, saved.y);
      if (Math.abs(window.scrollY - saved.y) > 2 && ++tries < 20) setTimeout(go, 150);
    }
    if (document.readyState === "complete") go(); else window.addEventListener("load", go);
  })();

  // ---- restart ----------------------------------------------------------------------------------
  // A banner pinned to the top of the window while the server restarts (it does not move the
  // page).
  function restarting(text) {
    var b = document.getElementById("restart-banner");
    if (!b) { b = document.createElement("div"); b.id = "restart-banner"; b.className = "restart-banner"; b.setAttribute("role", "status"); b.setAttribute("aria-live", "polite"); document.body.appendChild(b); }
    b.textContent = text || "Restarting\u2026 the page reloads (at the same place) when the server answers again.";
    return b;
  }
  // Restart buttons back to normal (the restart was refused or did not finish).
  function restartButtons() {
    document.querySelectorAll("[data-restart]").forEach(function (b) {
      if (b.dataset.label !== undefined) { b.textContent = b.dataset.label; delete b.dataset.label; }
      b.disabled = false;
    });
  }
  function restartFailed(text) {
    restarting(text).classList.add("gave-up");
    restartButtons();
  }
  // Wait for the next registry generation, then reload in place; gives up after 2 minutes.
  function waitRestart(gen, epoch) {
    var started = Date.now();
    function again() {
      if (Date.now() - started > RESTART_GIVE_UP_MS) {
        restartFailed("Still restarting after 2 minutes. Check the server log (or the service) and reload this page by hand.");
        return;
      }
      PrismPoll.fetch("/v1/config", {cache: "no-store", headers: {"Accept": "application/json"}}).then(json).then(function (j) {
        if (j.generation !== gen || j.epoch !== epoch) reloadKeep(null, true); else setTimeout(again, 700);
      }, function () { setTimeout(again, 1000); });
    }
    setTimeout(again, 700);
  }
  function afterRestart() { waitRestart(last ? last.generation : null, last ? last.epoch : null); }
  // Restart (existing POST /config/restart flow), then reload once the next generation answers.
  function restart(btn) {
    if (!confirm("Restart the server now? It reloads the config file and recompiles the enabled models.")) return;
    var gen = last ? last.generation : null, epoch = last ? last.epoch : null;
    if (btn) { if (btn.dataset.label === undefined) btn.dataset.label = btn.textContent; btn.disabled = true; btn.textContent = "Restarting…"; }
    restarting();
    var done = function () { waitRestart(gen, epoch); };
    fetch("/config/restart", {method: "POST", headers: {"Accept": "application/json"}}).then(function (r) {
      // Refused (403: a rejected origin or host name, ...): nothing restarts. A 401 (signed out)
      // never gets here: the fetch wrapper goes to the login page.
      if (r.status >= 400 && r.status < 500) {
        return r.json().catch(function () { return {}; }).then(function (j) {
          restartFailed("Restart refused (HTTP " + r.status + ")" + (j && j.message ? ": " + j.message : "."));
        });
      }
      done();
    }, done);
  }
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("[data-restart]");
    if (!b || b.type === "submit") return;
    e.preventDefault();
    restart(b);
  });

  // ---- form helpers -------------------------------------------------------------------------
  // A control differs from what the page last showed from the server (its default state).
  function edited(el) {
    if (el.type === "checkbox" || el.type === "radio") return el.checked !== el.defaultChecked;
    if (el.tagName === "SELECT") {
      for (var i = 0; i < el.options.length; i++) if (el.options[i].selected !== el.options[i].defaultSelected) return true;
      return false;
    }
    if (el.type === "number") {
      var a = el.value.trim(), b = el.defaultValue.trim();
      if (a === "" || b === "") return a !== b;
      return parseFloat(a) !== parseFloat(b);
    }
    return el.value !== el.defaultValue;
  }
  function current(el) {
    if (el.type === "checkbox") return el.checked;
    return el.value;
  }
  function baseline(el) {
    if (el.type === "checkbox") return el.defaultChecked;
    if (el.tagName === "SELECT") {
      for (var i = 0; i < el.options.length; i++) if (el.options[i].defaultSelected) return el.options[i].value;
      return el.options.length ? el.options[0].value : "";
    }
    return el.defaultValue;
  }
  function same(el, a, b) {
    if (el.type === "checkbox") return !!a === !!b;
    a = a === null || a === undefined ? "" : String(a); b = b === null || b === undefined ? "" : String(b);
    if (el.type === "number" && a.trim() !== "" && b.trim() !== "") return parseFloat(a) === parseFloat(b);
    return a === b;
  }
  // Set value and default (the new "shown by the server" state).
  function set(el, v) {
    if (el.type === "checkbox") { el.checked = !!v; el.defaultChecked = !!v; return; }
    v = v === null || v === undefined ? "" : String(v);
    if (el.tagName === "SELECT") {
      var found = false;
      for (var i = 0; i < el.options.length; i++) {
        var o = el.options[i], on = o.value === v; o.selected = on; o.defaultSelected = on; if (on) found = true;
      }
      if (!found) { var n = new Option(v + " — not a known option", v, true, true); el.add(n); }
      return;
    }
    el.value = v; el.defaultValue = v;
  }
  function flash(el) {
    if (!el) return;
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  }
  function note(el, text) {
    var holder = el.closest("label, td") || el.parentNode, n = holder.querySelector(".live-note");
    if (!text) { if (n) n.remove(); el.classList.remove("edited-conflict"); return; }
    if (!n) { n = document.createElement("span"); n.className = "live-note"; holder.appendChild(n); }
    n.textContent = text; el.classList.add("edited-conflict");
  }
  // Refresh one control to the server value `v`: unedited -> take it (flash when it changed);
  // edited -> keep the user's value; mark it when the server changed it too (saving writes the
  // user's value: last write wins). Returns true when the control now shows the server value.
  function refresh(el, v, describe) {
    if (!edited(el)) {
      var changed = !same(el, current(el), v);
      set(el, v); note(el, null); el.classList.remove("edited");
      if (changed) flash(el);
      return true;
    }
    if (same(el, current(el), v)) { set(el, v); note(el, null); el.classList.remove("edited"); return true; }
    if (!same(el, baseline(el), v)) note(el, "Changed elsewhere to " + (describe ? describe(v) : String(v)) + "; saving replaces it with your value.");
    else note(el, null);
    return false;
  }
  function markEdits(form) {
    form.addEventListener("input", function (e) { if (e.target.name) e.target.classList.toggle("edited", edited(e.target)); });
    form.addEventListener("change", function (e) { if (e.target.name) e.target.classList.toggle("edited", edited(e.target)); });
  }

  // ---- view state across re-renders ------------------------------------------------------------
  // Pages that poll re-render tables; without care every poll resets what the user did with
  // them. `keep(root, sig, render)` skips the render when the section's data signature `sig` is
  // unchanged, defers it while the user is busy in it (selecting text, pressing the mouse, e.g.
  // dragging a scrollbar, pointing at a chart), and otherwise carries over, by stable key, the
  // scroll offsets of every `.scroll` wrapper (and any element with `data-keep`), the open state
  // of every <details>, each table's sort column and direction, and the focused control.
  // Keys: `data-keep`, else the id, else the element's kind and position in the section.
  // `patch(box, items)` does the same per child (keyed cards): only changed children are
  // rebuilt, in place.
  var pointerIn = null;
  document.addEventListener("pointerdown", function (e) { pointerIn = e.target; }, true);
  ["pointerup", "pointercancel", "dragend"].forEach(function (t) { document.addEventListener(t, function () { pointerIn = null; }, true); });
  var KEPT = ".scroll, details, table, [data-keep]";
  function keyed(root) {
    var out = {}, seen = {};
    root.querySelectorAll(KEPT).forEach(function (e) {
      var k = e.dataset.keep || (e.id ? "#" + e.id : null);
      if (!k) { var kind = e.tagName.toLowerCase() + (e.classList.contains("scroll") ? ".scroll" : ""); seen[kind] = (seen[kind] || 0) + 1; k = kind + ":" + seen[kind]; }
      if (!(k in out)) out[k] = e;
    });
    return out;
  }
  function busyIn(root) {
    if (pointerIn && root.contains(pointerIn)) return true;
    var s = window.getSelection && window.getSelection();
    if (s && !s.isCollapsed && s.rangeCount && root.contains(s.getRangeAt(0).commonAncestorContainer)) return true;
    return !!root.querySelector("svg:hover, .keep-hover:hover");
  }
  function capture(root) {
    var st = {els: {}, focus: null};
    var map = keyed(root);
    Object.keys(map).forEach(function (k) {
      var e = map[k], v = {};
      if (e.scrollLeft || e.scrollTop) { v.left = e.scrollLeft; v.top = e.scrollTop; }
      if (e.tagName === "DETAILS") v.open = e.open;
      if (e.tagName === "TABLE") {
        var hs = e.tHead ? e.tHead.rows[0].cells : [];
        for (var i = 0; i < hs.length; i++) if (hs[i].dataset.dir) v.sort = [i, hs[i].dataset.dir];
      }
      st.els[k] = v;
    });
    var a = document.activeElement;
    if (a && a !== document.body && root.contains(a)) {
      var all = root.querySelectorAll(a.tagName);
      st.focus = {tag: a.tagName, name: a.name || null, text: a.textContent, index: Array.prototype.indexOf.call(all, a)};
    }
    return st;
  }
  // Sort a table's body by column `col` (numeric when the cells carry data-v).
  function sortTable(table, col, asc) {
    var tb = table.tBodies[0]; if (!tb) return;
    var rows = Array.prototype.slice.call(tb.rows);
    table.querySelectorAll("th").forEach(function (h) { delete h.dataset.dir; h.removeAttribute("aria-sort"); });
    var th = table.tHead && table.tHead.rows[0].cells[col]; if (th) { th.dataset.dir = asc ? "asc" : "desc"; th.setAttribute("aria-sort", asc ? "ascending" : "descending"); }
    rows.sort(function (a, b) {
      var x = a.cells[col], y = b.cells[col]; if (!x || !y) return 0;
      var vx = x.dataset.v, vy = y.dataset.v;
      if (vx !== undefined && vy !== undefined) { vx = parseFloat(vx); vy = parseFloat(vy); if (isNaN(vx)) vx = Infinity; if (isNaN(vy)) vy = Infinity; return asc ? vx - vy : vy - vx; }
      return asc ? x.textContent.localeCompare(y.textContent) : y.textContent.localeCompare(x.textContent);
    });
    rows.forEach(function (r) { tb.appendChild(r); });
  }
  function restore(root, st) {
    var map = keyed(root);
    Object.keys(st.els).forEach(function (k) {
      var e = map[k], v = st.els[k]; if (!e) return;
      if (v.sort) sortTable(e, v.sort[0], v.sort[1] === "asc");
      if (v.open !== undefined && e.open !== v.open) e.open = v.open;
    });
    // Scroll after every <details> is open again (closed ones have no layout).
    Object.keys(st.els).forEach(function (k) {
      var e = map[k], v = st.els[k]; if (!e || v.left === undefined) return;
      e.scrollLeft = v.left; e.scrollTop = v.top;
    });
    var f = st.focus;
    if (f && (!document.activeElement || document.activeElement === document.body)) {
      var cands = Array.prototype.filter.call(root.querySelectorAll(f.tag), function (e) { return (e.name || null) === f.name && e.textContent === f.text; });
      var pick = cands.length ? cands[0] : root.querySelectorAll(f.tag)[f.index];
      if (pick && pick.focus) try { pick.focus({preventScroll: true}); } catch (e) { pick.focus(); }
    }
  }
  // Re-render `root` with `render()` keeping its view state. `sig` (any string; undefined =
  // always render) skips unchanged data. Returns true when it rendered; false when skipped or
  // deferred (a deferred render is retried with the next call, its signature not taken).
  function keep(root, sig, render) {
    if (!root) return false;
    if (sig !== undefined && root._keepSig === sig) return false;
    if (root._keepSig !== undefined && busyIn(root)) return false;
    var st = capture(root);
    render();
    restore(root, st);
    root._keepSig = sig;
    return true;
  }
  // Keyed children of `box`: items [{key, sig, build}] in order; `build()` returns the new
  // element. Unchanged children (same sig) are left alone, changed ones replaced in place with
  // their view state, missing ones added, the rest removed.
  function patch(box, items) {
    var old = {};
    Array.prototype.forEach.call(box.children, function (c) { if (c.dataset.patchKey !== undefined) old[c.dataset.patchKey] = c; });
    var prev = null;
    items.forEach(function (it) {
      var cur = old[it.key]; delete old[it.key];
      if (!cur || (cur._keepSig !== it.sig && !busyIn(cur))) {
        var n = it.build(); n.dataset.patchKey = it.key; n._keepSig = it.sig;
        if (cur) { var st = capture(cur); cur.replaceWith(n); restore(n, st); }
        cur = n;
      }
      var want = prev ? prev.nextSibling : box.firstChild;
      if (want !== cur) box.insertBefore(cur, want);
      prev = cur;
    });
    Object.keys(old).forEach(function (k) { old[k].remove(); });
  }

  // ---- tabs ---------------------------------------------------------------------------------
  // `<nav class="tabs" role="tablist">` with `<a role="tab" href="#panel">` links: one panel
  // shows at a time, the chosen one in the URL's #fragment. Any #anchor inside a panel opens
  // that panel first and then scrolls to the anchor, so deep links keep working. Without
  // JavaScript every panel is in the page, stacked.
  function tabs(nav) {
    if (!nav) return null;
    if (nav._tabs) return nav._tabs;
    var links = Array.prototype.slice.call(nav.querySelectorAll('a[role="tab"]'));
    var panels = links.map(function (a) { return document.getElementById((a.getAttribute("href") || "").slice(1)); });
    if (!links.length || panels.some(function (p) { return !p; })) return null;
    links.forEach(function (a, i) {
      if (!a.id) a.id = panels[i].id + "-tab";
      a.setAttribute("aria-controls", panels[i].id);
      panels[i].setAttribute("role", "tabpanel"); panels[i].setAttribute("aria-labelledby", a.id);
    });
    var current = null;
    function indexOf(id) {
      if (!id) return null;
      var t = document.getElementById(id); if (!t) return null;
      for (var i = 0; i < panels.length; i++) if (panels[i] === t || panels[i].contains(t)) return i;
      return null;
    }
    function select(i) {
      links.forEach(function (a, k) { var on = k === i; a.setAttribute("aria-selected", on ? "true" : "false"); a.tabIndex = on ? 0 : -1; panels[k].hidden = !on; });
      current = i;
    }
    function fromHash() {
      var id = location.hash.slice(1), i = indexOf(id);
      if (i === null) { if (current === null) select(0); return; }
      select(i);
      var t = document.getElementById(id);
      if (t && t !== panels[i]) t.scrollIntoView(); else nav.scrollIntoView();
    }
    links.forEach(function (a, i) {
      a.addEventListener("click", function (e) {
        e.preventDefault(); select(i);
        if (history.replaceState) history.replaceState(null, "", a.getAttribute("href")); else location.hash = a.getAttribute("href");
      });
      a.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (!d) return;
        e.preventDefault(); var n = (i + d + links.length) % links.length; links[n].click(); links[n].focus();
      });
    });
    window.addEventListener("hashchange", fromHash);
    fromHash();
    nav._tabs = {
      select: select,
      current: function () { return current; },
      // Show the panel that holds the element `id` (a panel id or any anchor inside one).
      show: function (id) { var i = indexOf(id); if (i !== null) select(i); return i !== null; },
      indexOf: indexOf
    };
    return nav._tabs;
  }
  document.addEventListener("DOMContentLoaded", function () { document.querySelectorAll("nav.tabs").forEach(tabs); });

  // ---- forms without leaving the page ----------------------------------------------------------
  // `<form data-ajax>`: submitted with fetch (Accept: application/json) instead of navigating,
  // so the page keeps its scroll position; the clicked button is disabled while the request
  // runs and the answer is shown next to it. The page then updates the section in place: it
  // gets a "prism:action" event on the form ({ok, json, button}). Without JS the form posts
  // normally (its action carries the section's #fragment). `data-confirm` on a button asks
  // first.
  function inlineMsg(form, by) {
    var at = (by && by.closest(".actions, td, .inline-row")) || form, m = at.querySelector(".form-msg, .action-msg");
    if (!m) { m = document.createElement("span"); m.className = "action-msg"; m.setAttribute("aria-live", "polite"); at.appendChild(m); }
    return m;
  }
  function say(m, cls, text) { m.className = (m.classList.contains("form-msg") ? "form-msg " : "action-msg ") + cls; m.textContent = text; }
  function submitAjax(form, by) {
    var data = new FormData(form);
    if (by && by.name) data.append(by.name, by.value);
    var url = (by && by.getAttribute("formaction")) || form.getAttribute("action") || location.pathname;
    url = url.split("#")[0];
    var m = inlineMsg(form, by), label = by ? by.textContent : null;
    if (by) by.disabled = true;
    say(m, "muted", "working…");
    var fire = function (ok, j) { form.dispatchEvent(new CustomEvent("prism:action", {bubbles: true, detail: {ok: ok, json: j, button: by, message: m}})); };
    return fetch(url, {method: "POST", body: new URLSearchParams(data), headers: {"Accept": "application/json"}})
      .then(function (r) {
        return r.json().then(function (j) { j._status = r.status; return j; }, function () {
          return {success: false, _status: r.status, message: "the server answered HTTP " + r.status + (r.statusText ? " " + r.statusText : "") + (r.status === 403 ? " (not allowed from here)" : "")};
        });
      })
      .then(function (j) {
        if (by) { by.disabled = false; by.textContent = label; }
        say(m, j.success ? "ok-text" : "error-text", j.message || (j.success ? "done" : "failed"));
        poll(true);
        fire(!!j.success, j);
      })
      .catch(function (e) { if (by) by.disabled = false; say(m, "error-text", "failed: the server did not answer (" + e + "); it may be restarting or stopped, so check that it runs and try again."); fire(false, null); });
  }
  document.addEventListener("submit", function (e) {
    var form = e.target;
    if (e.defaultPrevented || !form.matches || !form.matches("form[data-ajax]")) return;
    var by = e.submitter;
    e.preventDefault();
    if (by && by.hasAttribute("data-restart")) { restart(by); return; }
    submitAjax(form, by);
  });

  // ---- native POST forms -------------------------------------------------------------------
  // A form that posts by navigating (Accounts, Collect settings, the Train queue, ...) would land on
  // the login page and lose what was typed when the session has ended. So when the polls last saw
  // it signed out, or saw no signed-in answer for SESSION_FRESH_MS (a tab that slept), the submit
  // waits for one poll-style GET /v1/config first: signed in, the form goes on with the same
  // button (`requestSubmit(submitter)` keeps its name, value and formaction); signed out, the
  // sign-in dialog opens and the form goes on after it. This runs first (capture on window) and
  // keeps the first submit from every other listener, so a `data-confirm` on submit asks once,
  // on the submit that goes on; one on the button's click has asked already. Skipped: forms with
  // `data-no-auth-check` (Sign out, the dialog's own), the /login and /setup pages, and pages
  // never signed in (refusal pages).
  var SESSION_FRESH_MS = 60000;
  function postForm(f) { return !!(f && f.getAttribute && String(f.getAttribute("method") || "").toLowerCase() === "post"); }
  function everSignedIn() {
    if (fullAnswer(last)) return true;
    var s = document.querySelector(".signed-in");
    return !!(s && String(s.textContent || "").trim());
  }
  function resubmit(form, by) {
    form._prismPass = true;
    try {
      if (typeof form.requestSubmit !== "function") form.submit();
      else if (!by) form.requestSubmit();
      else try { form.requestSubmit(by); } catch (e) { form.requestSubmit(); }
    } finally { form._prismPass = false; }
  }
  // Resolves true when signed in, false when signed out; rejects when the server does not answer.
  function sessionCheck() {
    if (signedOut) return Promise.resolve(false);
    return PrismPoll.fetch("/v1/config", {headers: {"Accept": "application/json"}, cache: "no-store"}).then(json).then(function (j) {
      if (!fullAnswer(j)) { signedOut = true; return false; }
      lastGood = Date.now();
      return true;
    });
  }
  window.addEventListener("submit", function (e) {
    var form = e.target;
    if (!form || form._prismPass || !postForm(form) || form.hasAttribute("data-no-auth-check")) return;
    if (location.pathname === "/login" || location.pathname === "/setup") return;
    if (!signedOut && Date.now() - lastGood < SESSION_FRESH_MS) return;
    if (!everSignedIn()) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    var by = e.submitter || null;
    sessionCheck().then(function (ok) {
      if (ok) { resubmit(form, by); return; }
      PrismPoll.signIn().then(function () {
        signedOut = false; lastGood = Date.now();
        resubmit(form, by);
        poll(true).then(schedule);
      });
    }, function () {
      // No answer: the browser posts as it would have (and shows what it gets).
      resubmit(form, by);
    });
  }, true);

  // ---- the restart page (message.html) ---------------------------------------------------------
  // `#restart-status[data-generation]`: poll until the next generation serves, then go to
  // `data-target` (a path on this server; anything else goes home). Gives up after 2 minutes.
  function restartPage() {
    var status = document.getElementById("restart-status");
    if (!status || status.dataset.generation === undefined) return;
    var target = status.dataset.target || "/", gen = Number(status.dataset.generation), started = Date.now();
    if (!/^\/(?![\/\\])/.test(target)) target = "/";
    function tick() {
      var secs = Math.round((Date.now() - started) / 1000);
      if (Date.now() - started > RESTART_GIVE_UP_MS) {
        status.textContent = "The server has not come back after " + secs + " s. Check the server log (or the service) and reload this page by hand.";
        status.className = "error";
        return;
      }
      PrismPoll.fetch("/v1/config", {cache: "no-store", headers: {"Accept": "application/json"}}).then(json).then(function (j) {
        if (j.generation === gen) throw failure("same generation");
        status.textContent = "Server is back; reloading…";
        PrismPoll.navigate(target);
      }).catch(function () {
        status.textContent = "Waiting for the server… (" + secs + " s)";
        setTimeout(tick, 1000);
      });
    }
    setTimeout(tick, 1500);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", restartPage); else restartPage();

  window.PrismLive = {
    onConfig: function (fn) { listeners.push(fn); if (last) fn(last, null); },
    poll: poll,
    restart: restart,
    restarting: restarting,
    afterRestart: afterRestart,
    reloadKeep: reloadKeep,
    submit: submitAjax,
    last: function () { return last; },
    keep: keep,
    patch: patch,
    sortTable: sortTable,
    scrollCue: scrollCue,
    tabs: tabs,
    view: {capture: capture, restore: restore, busy: busyIn},
    ui: {edited: edited, current: current, baseline: baseline, same: same, set: set, flash: flash, note: note, refresh: refresh, markEdits: markEdits}
  };
  poll().then(schedule);
  window.addEventListener("focus", function () { if (!document.hidden) poll(); });
  // Back from the back-forward cache, or the network is back: ask at once (poll() joins one in
  // flight, so never two).
  window.addEventListener("pageshow", function (e) { if (e && e.persisted && !document.hidden) poll().then(schedule); });
  window.addEventListener("online", function () { if (!document.hidden) poll().then(schedule); });
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) { clearTimeout(timer); timer = null; } else poll().then(schedule);
  });
})();
