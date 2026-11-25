//! `PrismTime` (`assets/time.js`): every time the pages show, in the configured zone and in the
//! style of the server's `Zone::display`. Run in `node` with fixed instants: IANA zones across a
//! DST change, `UTC`, fixed offsets, an unknown zone (the browser's), bad input, file stamps and
//! the training "done at" time.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).
#![allow(clippy::unwrap_used)]

use crate::js_harness::{asset, have_node, run_node_ok};
use std::ffi::OsStr;

#[test]
fn prism_time_formats_in_the_zone() {
    if !have_node() {
        return;
    }
    let cases = parity_cases();
    run_node_ok(DRIVER, &[asset("time.js").as_os_str(), OsStr::new(&cases)]);
}

/// QA-111: the server's `Zone::display` and the abbreviation map it writes on the page, for
/// zones whose en-US `Intl` name has no letters (`GMT+1`). The driver must print the same text.
/// Instants are relative to now (the map covers the next year), a half year apart, so both
/// daylight saving states are met; no text of the process-wide zone is used.
fn parity_cases() -> String {
    use blue_onyx_prism::clock::Zone;
    let now = chrono::Utc::now();
    let mut cases = Vec::new();
    for name in [
        "Europe/Berlin",
        "Europe/London",
        "Asia/Kolkata",
        "Australia/Sydney",
        "America/Sao_Paulo",
        "America/Los_Angeles",
        "Asia/Dubai",
        "UTC",
        "+05:30",
    ] {
        let zone: Zone = name.parse().unwrap();
        for days in [0, 91, 182, 273] {
            let t = now + chrono::TimeDelta::days(days);
            cases.push(serde_json::json!({
                "tz": zone.browser_name(),
                "abbr": zone.abbreviation_map(now),
                "ms": t.timestamp_millis(),
                "expected": zone.display(t),
            }));
        }
    }
    serde_json::to_string(&cases).unwrap()
}

const DRIVER: &str = r#"
const assert = require('assert');
require(process.argv[2]);
const T = globalThis.PrismTime;
const LA = 'America/Los_Angeles';
const winter = '2026-11-16T22:02:11Z';   // PST
const summer = '2026-07-04T21:02:11Z';   // PDT
assert.strictEqual(T.zone(), '');
assert.strictEqual(T.when(winter, LA), '2026-11-16 14:02 PST');
assert.strictEqual(T.when(summer, LA), '2026-07-04 14:02 PDT');
assert.strictEqual(T.when(Date.parse(winter), LA), '2026-11-16 14:02 PST');
assert.strictEqual(T.when('2026-11-16T23:02:11+01:00', LA), '2026-11-16 14:02 PST');
assert.strictEqual(T.date(winter, LA), '2026-11-16');
assert.strictEqual(T.date('2026-11-17T03:00:00Z', LA), '2026-11-16');
assert.strictEqual(T.day(winter, LA), 'Nov 16 14:02');
assert.strictEqual(T.clock(Date.parse(winter), LA), '14:02:11');
assert.strictEqual(T.stamp(Date.parse(winter), LA), '20261116-140211');
// Midnight is 00, never 24.
assert.strictEqual(T.clock(Date.parse('2026-11-16T08:00:05Z'), LA), '00:00:05');
assert.strictEqual(T.when('2026-11-16T22:02:11Z', 'Europe/Berlin'), '2026-11-16 23:02 +01:00'); // no map: the numeric offset
assert.strictEqual(T.date('2026-11-16T23:30:00Z', 'Europe/Berlin'), '2026-11-17');
assert.strictEqual(T.when(winter, 'UTC'), '2026-11-16 22:02 UTC');
assert.strictEqual(T.when(winter, '+05:30'), '2026-11-17 03:32 +05:30');
assert.strictEqual(T.when(winter, '-08:00'), '2026-11-16 14:02 -08:00');
assert.strictEqual(T.stamp(Date.parse(winter), '+05:30'), '20261117-033211');
assert.strictEqual(T.day(winter, '-08:00'), 'Nov 16 14:02');
// A zone without letters (Intl says GMT+4) gets the numeric offset.
assert.strictEqual(T.when(winter, 'Asia/Dubai'), '2026-11-17 02:02 +04:00');
// An unknown zone falls back to the browser's: still a well-formed time.
assert.ok(/^\d{4}-\d\d-\d\d \d\d:\d\d/.test(T.when(winter, 'Not/AZone')), T.when(winter, 'Not/AZone'));
assert.ok(/^\d{4}-\d\d-\d\d \d\d:\d\d/.test(T.when(winter, '')));
// Bad input comes back unchanged.
assert.strictEqual(T.when('garbage', LA), 'garbage');
assert.strictEqual(T.when(null, LA), '');
assert.strictEqual(T.when(undefined, LA), '');
assert.strictEqual(T.clock('', LA), '');
assert.strictEqual(T.date('nope', LA), 'nope');
// "done at": time of day, plus the weekday in the zone when over 20 h away.
const now = Date.parse(winter); // Mon 2026-11-16 14:02 PST
assert.strictEqual(T.doneAt(3600, now, LA), '15:02');
assert.strictEqual(T.doneAt(21 * 3600, now, LA), 'Tue 11:02');
// Berlin is 23:02 Monday at that instant.
assert.strictEqual(T.doneAt(21 * 3600, now, 'Europe/Berlin'), 'Tue 20:02');
// Weekday follows the zone, not UTC: 07:00Z is still Monday 23:00 in Los Angeles.
assert.strictEqual(T.doneAt(21 * 3600, Date.parse('2026-11-17T07:00:00Z'), LA), 'Tue 20:00');
assert.strictEqual(T.doneAt(21 * 3600, Date.parse('2026-11-17T07:00:00Z'), 'UTC'), 'Wed 04:00');
// The page's zone is read lazily from the root element.
globalThis.document = { documentElement: { dataset: { tz: LA } } };
assert.strictEqual(T.zone(), LA);
assert.strictEqual(T.when(winter), '2026-11-16 14:02 PST');
// QA-111: with the server's map, the letters the server prints.
assert.strictEqual(T.when('2026-11-16T22:02:11Z', 'Europe/Berlin', '60:CET,120:CEST'), '2026-11-16 23:02 CET');
assert.strictEqual(T.when('2026-07-16T22:02:11Z', 'Europe/Berlin', '60:CET,120:CEST'), '2026-07-17 00:02 CEST');
// Every zone of the Rust side prints the same text (`Zone::display`).
for (const c of JSON.parse(process.argv[3])) {
  const got = T.when(c.ms, c.tz, c.abbr);
  assert.strictEqual(got, c.expected, JSON.stringify(c));
}
console.log('ok');
"#;
