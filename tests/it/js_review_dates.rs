//! The Review tab's date range presets (`dateRange`, `datePreset` in `assets/review-rules.js`):
//! month, year and leap-day edges, and that a preset's range names the preset back.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).
#![allow(clippy::unwrap_used)]

use crate::js_harness::{asset, have_node, run_node_ok};

const DRIVER: &str = r#"
const assert = require('assert');
require(process.argv[2]);
const R = globalThis.ReviewRules;

assert.deepStrictEqual(R.dateRange('any', '2026-11-26'), { from: '', to: '' });
assert.deepStrictEqual(R.dateRange('last7', '2026-11-26'), { from: '2026-11-20', to: '2026-11-26' });
assert.deepStrictEqual(R.dateRange('last30', '2026-11-26'), { from: '2026-10-28', to: '2026-11-26' });
assert.deepStrictEqual(R.dateRange('older30', '2026-11-26'), { from: '', to: '2026-10-27' });
assert.deepStrictEqual(R.dateRange('nonsense', '2026-11-26'), { from: '', to: '' });
// Month and year edges.
assert.deepStrictEqual(R.dateRange('last7', '2026-01-03'), { from: '2025-12-28', to: '2026-01-03' });
assert.deepStrictEqual(R.dateRange('last30', '2026-03-01'), { from: '2026-01-31', to: '2026-03-01' });
// Leap years: 2024 has a 29 February, 2026 and 2100 do not.
assert.deepStrictEqual(R.dateRange('last7', '2024-03-02'), { from: '2024-02-25', to: '2024-03-02' });
assert.deepStrictEqual(R.dateRange('last7', '2026-03-02'), { from: '2026-02-24', to: '2026-03-02' });
assert.strictEqual(R.dateRange('older30', '2024-03-30').to, '2024-02-29');
assert.strictEqual(R.dateRange('older30', '2100-03-31').to, '2100-03-01');
assert.strictEqual(R.dateRange('older30', '0100-01-05').to, '0099-12-06');

// The inverse, and the round trip for every preset on edge days.
for (const today of ['2026-11-26', '2026-01-03', '2024-03-01', '2025-12-31']) {
  for (const p of ['any', 'last7', 'last30', 'older30']) {
    const r = R.dateRange(p, today);
    assert.strictEqual(R.datePreset(r.from, r.to, today), p, p + ' ' + today);
  }
}
assert.strictEqual(R.datePreset('2026-11-01', '2026-11-05', '2026-11-26'), 'custom');
assert.strictEqual(R.datePreset('2026-11-20', '', '2026-11-26'), 'custom');
assert.strictEqual(R.datePreset('', '2026-11-26', '2026-11-26'), 'custom');
// A range of another day is custom, even with the same width.
assert.strictEqual(R.datePreset('2026-11-19', '2026-11-25', '2026-11-26'), 'custom');
// Missing values count as unbounded.
assert.strictEqual(R.datePreset(undefined, null, '2026-11-26'), 'any');
console.log('ok');
"#;

#[test]
fn date_presets_and_their_inverse() {
    if !have_node() {
        return;
    }
    run_node_ok(DRIVER, &[asset("review-rules.js").as_os_str()]);
}
