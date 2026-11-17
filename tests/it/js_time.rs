//! `PrismTime` (`assets/time.js`): every time the pages show, in the configured zone and in the
//! style of the server's `Zone::display`. Run in `node` with fixed instants: IANA zones across a
//! DST change, `UTC`, fixed offsets, an unknown zone (the browser's), bad input, file stamps and
//! the training "done at" time.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

fn have_node() -> bool {
    let ok = Command::new("node")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        assert!(
            std::env::var("BOP_REQUIRE_NODE").as_deref() != Ok("1"),
            "BOP_REQUIRE_NODE=1 but `node` does not run: install Node or unset the variable"
        );
        println!("skipped: no node");
    }
    ok
}

#[test]
fn prism_time_formats_in_the_zone() {
    if !have_node() {
        return;
    }
    let time_js = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/time.js");
    let dir = std::env::temp_dir().join(format!("bop-js-time-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let driver = dir.join("driver.js");
    std::fs::write(&driver, DRIVER).unwrap();
    let out = Command::new("node")
        .arg(&driver)
        .arg(&time_js)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "node failed ({}): {}{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
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
assert.strictEqual(T.when('2026-11-16T22:02:11Z', 'Europe/Berlin').slice(0, 16), '2026-11-16 23:02');
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
console.log('ok');
"#;
