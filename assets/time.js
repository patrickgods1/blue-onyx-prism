// Blue Onyx Prism: every time shown to the user, in the configured time zone (`timezone` in the
// config; the server writes it on the root element as `data-tz`: an IANA name, `UTC`, a fixed
// offset such as `+08:00`, or empty = the browser's zone). Same style as the server's
// `Zone::display` (src/clock.rs): `2026-11-16 14:02 PST`; letters come from `data-tz-abbr`. Pure: no DOM beyond a lazy read of
// `document.documentElement.dataset.tz`; every function takes an optional zone for tests.
// Invalid input comes back unchanged ('' for null/empty).
(function () {
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  var cache = {};

  function zone() {
    try {
      var d = globalThis.document;
      return (d && d.documentElement && d.documentElement.dataset && d.documentElement.dataset.tz) || '';
    } catch (e) { return ''; }
  }
  function pad(n, w) { var s = String(n); while (s.length < (w || 2)) s = '0' + s; return s; }
  function offsetMinutes(tz) {
    var m = /^([+-])(\d{2}):?(\d{2})$/.exec(tz);
    if (!m) return null;
    var v = parseInt(m[2], 10) * 60 + parseInt(m[3], 10);
    return m[1] === '-' ? -v : v;
  }
  function formatter(tz, abbr) {
    var key = tz + '|' + (abbr ? 1 : 0);
    if (cache[key]) return cache[key];
    var o = { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23', weekday: 'short' };
    if (abbr) o.timeZoneName = 'short';
    if (tz) o.timeZone = tz;
    var f;
    try { f = new Intl.DateTimeFormat('en-US', o); } catch (e) {
      if (!tz) throw e;
      delete o.timeZone; // unknown zone: the browser's
      f = new Intl.DateTimeFormat('en-US', o);
    }
    cache[key] = f;
    return f;
  }
  function offsetLabel(min) {
    return (min < 0 ? '-' : '+') + pad(Math.floor(Math.abs(min) / 60)) + ':' + pad(Math.abs(min) % 60);
  }
  // The server's `data-tz-abbr` (`60:CET,120:CEST`, src/clock.rs `Zone::abbreviation_map`) as
  // {minutes: 'ABBR'}: Intl in en-US names no letters for most zones outside the US.
  function abbrMap(text) {
    var out = {};
    String(text || '').split(',').forEach(function (pair) {
      var m = /^(-?\d+):([A-Za-z0-9+-]+)$/.exec(pair);
      if (m) out[m[1]] = m[2];
    });
    return out;
  }
  function pageAbbr() {
    try {
      var d = globalThis.document;
      return (d && d.documentElement && d.documentElement.dataset && d.documentElement.dataset.tzAbbr) || '';
    } catch (e) { return ''; }
  }
  // Wall-clock fields of `ms` in `tz`: {y, mo, d, h, mi, s, wd (0 = Sunday), label}.
  // `abbr`: the zone's abbreviation text (default: the page's `data-tz-abbr`).
  function parts(ms, tz, abbr) {
    tz = tz === undefined ? zone() : tz || '';
    var fixed = offsetMinutes(tz);
    var f, shift = 0, label;
    if (fixed !== null) { f = formatter('UTC', false); shift = fixed * 60000; label = offsetLabel(fixed); }
    else if (tz === 'UTC') { f = formatter('UTC', false); label = 'UTC'; }
    else f = formatter(tz, true);
    var p = {};
    f.formatToParts(new Date(ms + shift)).forEach(function (x) { p[x.type] = x.value; });
    if (label === undefined) {
      // The zone's offset at this instant, from the wall clock Intl printed.
      var asUtc = Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour % 24, +p.minute, +p.second);
      var min = Math.round((asUtc - Math.floor(ms / 1000) * 1000) / 60000);
      var named = abbrMap(abbr === undefined ? pageAbbr() : abbr)[String(min)];
      var intl = p.timeZoneName || '';
      // The server's abbreviation, else Intl's letters, else the numeric offset (as the server
      // writes a zone with no letters).
      label = named || (intl && !/^(GMT|UTC)[+-]/.test(intl) ? intl : offsetLabel(min));
    }
    return {
      y: p.year, mo: +p.month, d: p.day, h: p.hour === '24' ? '00' : p.hour, mi: p.minute, s: p.second,
      wd: WEEKDAYS.indexOf(p.weekday), label: label
    };
  }
  function toMs(v) {
    if (v === null || v === undefined || v === '') return NaN;
    if (typeof v === 'number') return v;
    return new Date(v).getTime();
  }
  function wrap(v, fn) {
    var ms = toMs(v);
    if (isNaN(ms)) return v === null || v === undefined ? '' : String(v);
    try { return fn(ms); } catch (e) { return String(v); }
  }

  var T = {
    zone: zone,
    // 2026-11-16 14:02 PST
    when: function (v, tz, abbr) {
      return wrap(v, function (ms) {
        var p = parts(ms, tz, abbr);
        return p.y + '-' + pad(p.mo) + '-' + p.d + ' ' + p.h + ':' + p.mi + (p.label ? ' ' + p.label : '');
      });
    },
    // 2026-11-16
    date: function (v, tz) {
      return wrap(v, function (ms) { var p = parts(ms, tz); return p.y + '-' + pad(p.mo) + '-' + p.d; });
    },
    // Nov 16 14:02 (chart axes)
    day: function (v, tz) {
      return wrap(v, function (ms) { var p = parts(ms, tz); return MONTHS[p.mo - 1] + ' ' + parseInt(p.d, 10) + ' ' + p.h + ':' + p.mi; });
    },
    // 14:02:11
    clock: function (v, tz) {
      return wrap(v, function (ms) { var p = parts(ms, tz); return p.h + ':' + p.mi + ':' + p.s; });
    },
    // 20261116-140211 (file names)
    stamp: function (v, tz) {
      return wrap(v, function (ms) {
        var p = parts(ms, tz);
        return p.y + pad(p.mo) + p.d + '-' + p.h + p.mi + p.s;
      });
    },
    // "14:20", or "Sat 09:05" when `etaS` is over 20 h away: the wall-clock time `etaS` seconds after `now`.
    doneAt: function (etaS, now, tz) {
      var ms = (now === undefined || now === null ? Date.now() : now) + etaS * 1000;
      var p = parts(ms, tz);
      var t = p.h + ':' + p.mi;
      return etaS > 20 * 3600 ? WEEKDAYS[p.wd] + ' ' + t : t;
    }
  };
  globalThis.PrismTime = T;
})();
