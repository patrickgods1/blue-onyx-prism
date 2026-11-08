//! The light answer of `GET /v1/training?poll=1&have=...` that `assets/training.js` asks for every
//! few seconds: the epochs a page already holds are not sent again.
//!
//! `have` lists what the page holds, one `<run id>:<last epoch>:<prefix key>:<full key>` per run,
//! comma separated: the epoch of its last history row and the two keys the server sent with that
//! history (`history_keys`). A key is a digest of history rows ([`digests`]): the prefix key covers
//! every row but the last, the full key every row. The server finds the page's last epoch in its
//! own rows and answers, per run:
//!
//! - its rows up to that epoch digest to the full key: the page holds them all, so only the rows
//!   after it are sent (none for a run that did not change), with `history_since` = epoch + 1;
//! - else its rows before that epoch digest to the prefix key: only the last row changed (an eval
//!   event added its keys to the epoch's row), so it and the rows after it are sent, with
//!   `history_since` = that epoch;
//! - else (no such run on the page, a new attempt that restarted from epoch 1, the oldest rows
//!   dropped at [`crate::training::progress::MAX_HISTORY`], rows out of epoch order) the whole
//!   history, without `history_since`.
//!
//! The page keeps its rows before `history_since` and appends the ones sent
//! (`TrainMetrics.poll.merge` in `assets/train-metrics.js`, checked against [`answer`] by node in
//! the tests). Every answer carries the run's `history_keys` for the history it now stands for.
//!
//! The poll also leaves out what the page's tick never reads: `capability` (one full fetch on
//! load), `losses` (the history holds `loss`), the `log` of runs and tasks that are not queued,
//! running or paused (their log is on the rendered page) and the busy ones' lines past the last
//! [`POLL_LOG_LINES`], a run's install, benchmark and folder fields ([`UNREAD_RUN_FIELDS`]; of
//! `dataset` only `warnings`), the tasks' `result`, and the static `metrics` catalog once the page
//! sends `have` (its first poll has none and keeps the catalog). Without `poll` the answer is the
//! full one.

use crate::training::progress::EpochPoint;
use crate::training::service::RunRecord;
use crate::training::task::TaskRecord;
use serde_json::{Map, Value};
use std::collections::HashMap;

/// Log lines of a busy run or task in a poll answer (the page shows the last 40).
pub(super) const POLL_LOG_LINES: usize = 40;

/// Fields of a run the page's tick never reads, left out of a poll answer (the rendered page shows
/// them; they change only with a state the tick reloads the page for). Of `dataset` only
/// `warnings` stays.
const UNREAD_RUN_FIELDS: &[&str] = &[
    "output",
    "registered",
    "benchmark",
    "benchmark_error",
    "baseline",
    "run_dir",
];

/// Entries of `have` read (the rest are answered in full).
const MAX_HAVE: usize = 2000;

/// What the page holds of one run's history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Have {
    pub(super) last_epoch: u32,
    pub(super) prefix: u64,
    pub(super) full: u64,
}

/// `have` of the query: run id -> what the page holds. Entries that do not parse are left out (that
/// run is answered in full).
pub(super) fn parse_have(raw: &str) -> HashMap<String, Have> {
    raw.split(',')
        .filter(|e| !e.is_empty())
        .take(MAX_HAVE)
        .filter_map(|e| {
            let mut parts = e.rsplitn(4, ':');
            let full = u64::from_str_radix(parts.next()?, 16).ok()?;
            let prefix = u64::from_str_radix(parts.next()?, 16).ok()?;
            let last_epoch = parts.next()?.parse().ok()?;
            let id = parts.next().filter(|id| !id.is_empty())?;
            Some((
                id.to_string(),
                Have {
                    last_epoch,
                    prefix,
                    full,
                },
            ))
        })
        .collect()
}

/// FNV-1a, 64 bit: stable across builds and restarts, so a page's keys stay valid after a service
/// restart that reads the same history.
#[derive(Clone, Copy)]
struct Fnv(u64);

impl Fnv {
    const START: Self = Self(0xcbf2_9ce4_8422_2325);

    fn bytes(mut self, b: &[u8]) -> Self {
        for x in b {
            self.0 ^= u64::from(*x);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self
    }

    fn row(self, p: &EpochPoint) -> Self {
        let mut h = self
            .bytes(&p.epoch.to_le_bytes())
            .bytes(&p.elapsed_s.to_bits().to_le_bytes())
            .bytes(&(p.metrics.len() as u64).to_le_bytes());
        for (k, v) in &p.metrics {
            h = h
                .bytes(&(k.len() as u64).to_le_bytes())
                .bytes(k.as_bytes())
                .bytes(&v.to_bits().to_le_bytes());
        }
        h
    }
}

/// The digest of `rows[..i]` at index `i`, for every `i` in `0..=rows.len()`: element 0 stands for
/// no rows, the last for all of them.
fn digests(rows: &[EpochPoint]) -> Vec<u64> {
    let mut out = Vec::with_capacity(rows.len() + 1);
    let mut h = Fnv::START;
    out.push(h.0);
    for r in rows {
        h = h.row(r);
        out.push(h.0);
    }
    out
}

/// A key as the page sends it back.
fn hex(k: u64) -> String {
    format!("{k:016x}")
}

/// Where the rows to send start, given what the page holds: `Some(i)` = send `rows[i..]` with
/// `history_since` = the epoch the page keeps rows before; None = the whole history.
fn resume_at(rows: &[EpochPoint], d: &[u64], have: Option<&Have>) -> Option<(usize, u32)> {
    let have = have?;
    if !rows
        .windows(2)
        .all(|w| matches!(w, [a, b] if a.epoch < b.epoch))
    {
        return None;
    }
    let i = rows.iter().position(|r| r.epoch == have.last_epoch)?;
    if d.get(i + 1) == Some(&have.full) {
        Some((i + 1, have.last_epoch.saturating_add(1)))
    } else if d.get(i) == Some(&have.prefix) {
        Some((i, have.last_epoch))
    } else {
        None
    }
}

/// `r` as the poll answers it ([`answer`]).
fn run_value(mut r: RunRecord, have: Option<&Have>) -> Value {
    let rows = r
        .progress
        .as_mut()
        .map(|p| {
            p.losses.clear();
            std::mem::take(&mut p.history)
        })
        .unwrap_or_default();
    let busy = r.state.is_busy();
    if busy {
        let skip = r.log.len().saturating_sub(POLL_LOG_LINES);
        r.log.drain(..skip);
    }
    let mut v = serde_json::to_value(&r).unwrap_or(Value::Null);
    let Some(obj) = v.as_object_mut() else {
        return v;
    };
    if !busy {
        obj.remove("log");
    }
    for k in UNREAD_RUN_FIELDS {
        obj.remove(*k);
    }
    if let Some(d) = obj.get_mut("dataset").and_then(Value::as_object_mut) {
        d.retain(|k, _| k == "warnings");
    }
    if let Some(p) = obj.get_mut("progress").and_then(Value::as_object_mut) {
        p.remove("losses");
        let d = digests(&rows);
        let (from, since) = match resume_at(&rows, &d, have) {
            Some((i, since)) => (i, Some(since)),
            None => (0, None),
        };
        p.insert(
            "history".into(),
            serde_json::to_value(rows.get(from..).unwrap_or_default()).unwrap_or_default(),
        );
        if let Some(s) = since {
            p.insert("history_since".into(), Value::from(s));
        }
        // Keys of every row but the last, and of all of them (`d` has one more entry than rows).
        if let [.., prefix, full] = d.as_slice()
            && !rows.is_empty()
        {
            p.insert(
                "history_keys".into(),
                Value::from(vec![hex(*prefix), hex(*full)]),
            );
        }
    }
    v
}

/// `t` as the poll answers it: no `result`, a `log` only while it is busy (its tail).
fn task_value(t: &TaskRecord) -> Value {
    let mut v = serde_json::to_value(t).unwrap_or(Value::Null);
    if let Some(obj) = v.as_object_mut() {
        obj.remove("result");
        if t.state.is_busy() {
            let skip = t.log.len().saturating_sub(POLL_LOG_LINES);
            let tail = t.log.get(skip..).unwrap_or_default();
            obj.insert("log".into(), Value::from(tail.to_vec()));
        } else {
            obj.remove("log");
        }
    }
    v
}

/// The poll answer (see the module docs) of `runs` and `tasks` for a page holding `have`; None =
/// the query had no `have` (the page's first poll), so the answer carries the `metrics` catalog,
/// which the page keeps.
pub(super) fn answer(
    runs: Vec<RunRecord>,
    tasks: &[TaskRecord],
    busy: bool,
    metric_columns: Value,
    metrics: Value,
    have: Option<&HashMap<String, Have>>,
) -> Value {
    let none = HashMap::new();
    let held = have.unwrap_or(&none);
    let mut out = Map::new();
    out.insert("busy".into(), Value::from(busy));
    out.insert(
        "runs".into(),
        Value::from(
            runs.into_iter()
                .map(|r| {
                    let h = held.get(&r.id);
                    run_value(r, h)
                })
                .collect::<Vec<_>>(),
        ),
    );
    out.insert(
        "tasks".into(),
        Value::from(tasks.iter().map(task_value).collect::<Vec<_>>()),
    );
    out.insert("metric_columns".into(), metric_columns);
    if have.is_none() {
        out.insert("metrics".into(), metrics);
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::dataset::SceneFilter;
    use crate::training::progress::TrainProgress;
    use crate::training::service::RunState;

    fn row(epoch: u32, map: f64) -> EpochPoint {
        EpochPoint {
            epoch,
            elapsed_s: f64::from(epoch) * 10.0,
            metrics: [("map".to_string(), map), ("loss".to_string(), 1.0 / map)]
                .into_iter()
                .collect(),
        }
    }

    fn run_with(id: &str, rows: Vec<EpochPoint>) -> RunRecord {
        let mut r = crate::server::training::tests::record(
            id,
            SceneFilter::default(),
            None,
            RunState::Running,
            None,
        );
        r.log = (0..60).map(|i| format!("line {i}")).collect();
        r.progress = Some(TrainProgress {
            epoch: rows.last().map_or(0, |r| r.epoch),
            losses: vec![1.0; rows.len()],
            history: rows,
            ..TrainProgress::default()
        });
        r
    }

    fn poll(runs: Vec<RunRecord>, have: &str) -> Value {
        answer(
            runs,
            &[],
            true,
            Value::Null,
            Value::from("catalog"),
            Some(&parse_have(have)),
        )
    }

    /// `have` for what the page holds after `v` (the first run of an answer).
    fn have_of(id: &str, v: &Value) -> String {
        let p = &v["runs"][0]["progress"];
        let keys = p["history_keys"].as_array().unwrap();
        let last = p["history"].as_array().unwrap().last().unwrap()["epoch"].clone();
        format!(
            "{id}:{last}:{}:{}",
            keys[0].as_str().unwrap(),
            keys[1].as_str().unwrap()
        )
    }

    fn epochs(v: &Value) -> Vec<u64> {
        v["runs"][0]["progress"]["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["epoch"].as_u64().unwrap())
            .collect()
    }

    #[test]
    fn have_parses_ids_epochs_and_keys_and_skips_what_does_not() {
        let h = parse_have("a-1:3:00000000000000ff:10,,bad,x:y:1:2,b:c:7:a:b");
        assert_eq!(h.len(), 2, "{h:?}");
        assert_eq!(
            h["a-1"],
            Have {
                last_epoch: 3,
                prefix: 255,
                full: 16
            }
        );
        // An id may itself hold a colon: the three fields are taken from the right.
        assert_eq!(h["b:c"].last_epoch, 7);
    }

    #[test]
    fn a_page_with_no_history_gets_all_of_it_and_the_keys() {
        let v = poll(vec![run_with("r", vec![row(1, 0.1), row(2, 0.2)])], "");
        assert_eq!(epochs(&v), [1, 2]);
        let p = &v["runs"][0]["progress"];
        assert!(p.get("history_since").is_none());
        assert!(p.get("losses").is_none());
        assert_eq!(p["history_keys"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn an_unchanged_run_carries_no_rows_and_a_new_epoch_only_itself() {
        let rows = vec![row(1, 0.1), row(2, 0.2), row(3, 0.3)];
        let first = poll(vec![run_with("r", rows.clone())], "");
        let have = have_of("r", &first);
        let same = poll(vec![run_with("r", rows.clone())], &have);
        assert!(epochs(&same).is_empty());
        assert_eq!(same["runs"][0]["progress"]["history_since"], 4);
        assert_eq!(
            same["runs"][0]["progress"]["history_keys"],
            first["runs"][0]["progress"]["history_keys"]
        );
        let mut more = rows.clone();
        more.push(row(4, 0.4));
        let next = poll(vec![run_with("r", more)], &have);
        assert_eq!(epochs(&next), [4]);
        assert_eq!(next["runs"][0]["progress"]["history_since"], 4);
    }

    #[test]
    fn a_changed_last_row_is_sent_again_from_its_epoch() {
        let rows = vec![row(1, 0.1), row(2, 0.2)];
        let have = have_of("r", &poll(vec![run_with("r", rows.clone())], ""));
        let mut changed = rows;
        changed[1].metrics.insert("map50".into(), 0.5);
        changed.push(row(3, 0.3));
        let v = poll(vec![run_with("r", changed)], &have);
        assert_eq!(epochs(&v), [2, 3]);
        assert_eq!(v["runs"][0]["progress"]["history_since"], 2);
    }

    #[test]
    fn a_restart_a_trimmed_start_or_disorder_sends_everything() {
        let rows = vec![row(1, 0.1), row(2, 0.2), row(3, 0.3)];
        let have = have_of("r", &poll(vec![run_with("r", rows.clone())], ""));
        // A new attempt from epoch 1 that reached epoch 3 again with other values.
        let again = vec![row(1, 0.15), row(2, 0.25), row(3, 0.35)];
        let v = poll(vec![run_with("r", again)], &have);
        assert_eq!(epochs(&v), [1, 2, 3]);
        assert!(v["runs"][0]["progress"].get("history_since").is_none());
        // The oldest row dropped: the prefix no longer matches.
        let trimmed = vec![row(2, 0.2), row(3, 0.3), row(4, 0.4)];
        assert_eq!(
            epochs(&poll(vec![run_with("r", trimmed)], &have)),
            [2, 3, 4]
        );
        // Shorter than the page's (no row of its last epoch).
        let shorter = vec![row(1, 0.1)];
        assert_eq!(epochs(&poll(vec![run_with("r", shorter)], &have)), [1]);
        // Rows out of epoch order are never resumed.
        let odd = vec![row(1, 0.1), row(3, 0.3), row(2, 0.2), row(4, 0.4)];
        assert_eq!(epochs(&poll(vec![run_with("r", odd)], &have)).len(), 4);
    }

    #[test]
    fn a_task_keeps_its_log_tail_while_busy_and_never_its_result() {
        use crate::training::task::TaskKind;
        let mut t = TaskRecord {
            id: "t".into(),
            kind: TaskKind::Autolabel,
            title: "Label".into(),
            created: String::new(),
            state: RunState::Running,
            stage: None,
            step: 1,
            steps: 2,
            percent: 50,
            detail: String::new(),
            log: (0..50).map(|i| format!("l{i}")).collect(),
            summary: None,
            result: Some(serde_json::json!({ "frames": 3 })),
            error: None,
            finished: None,
            after: None,
            uses: Vec::new(),
        };
        let v = task_value(&t);
        assert_eq!(v["log"].as_array().unwrap().len(), POLL_LOG_LINES);
        assert_eq!(v["log"][0], "l10");
        assert!(v.get("result").is_none());
        t.state = RunState::Done;
        assert!(task_value(&t).get("log").is_none());
    }

    /// Whether `node` runs here; missing, a skip, or with `BOP_REQUIRE_NODE=1` (CI) a failure.
    fn have_node() -> bool {
        let ok = std::process::Command::new("node")
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

    /// `TrainMetrics.poll` of `assets/train-metrics.js`: merges `answer` into `store`, answers the
    /// new store, its `have` and the merged runs.
    const MERGE_DRIVER: &str = r#"
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({ window: {}, console });
vm.runInContext(fs.readFileSync(process.env.BOP_TRAIN_METRICS, 'utf8'), ctx);
const P = ctx.window.TrainMetrics.poll;
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const runs = P.merge(input.store, input.answer.runs);
process.stdout.write(JSON.stringify({ store: input.store, have: P.have(input.store), runs }));
"#;

    fn node_merge(store: &Value, answer: &Value) -> Value {
        use std::io::Write;
        let js = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/train-metrics.js");
        let mut child = std::process::Command::new("node")
            .args(["-e", MERGE_DRIVER])
            .env("BOP_TRAIN_METRICS", js)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let input = serde_json::json!({ "store": store, "answer": answer });
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.to_string().as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "node: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }

    #[test]
    fn the_page_merges_every_poll_back_to_the_full_history() {
        if !have_node() {
            return;
        }
        let mut changed_last = vec![row(1, 0.1), row(2, 0.2)];
        if let Some(last) = changed_last.last_mut() {
            last.metrics.insert("map50".into(), 0.4);
        }
        // Each step: the histories of runs "a" and "b" on the server (None = no such run).
        type History = Option<Vec<EpochPoint>>;
        let steps: Vec<(History, History)> = vec![
            (Some(vec![row(1, 0.1)]), None),
            (Some(vec![row(1, 0.1), row(2, 0.2)]), Some(vec![])),
            (Some(changed_last.clone()), Some(vec![row(1, 0.5)])),
            (Some(changed_last.clone()), Some(vec![row(1, 0.5)])),
            (
                Some({
                    let mut v = changed_last.clone();
                    v.push(row(3, 0.3));
                    v
                }),
                None,
            ),
            // A new attempt from epoch 1, and "b" back after it left the answer.
            (
                Some(vec![row(1, 0.9)]),
                Some(vec![row(1, 0.5), row(2, 0.6)]),
            ),
            (
                Some(vec![row(1, 0.9)]),
                Some(vec![row(1, 0.5), row(2, 0.6)]),
            ),
        ];
        let mut store = serde_json::json!({});
        let mut have = String::new();
        for (n, (a, b)) in steps.into_iter().enumerate() {
            let runs: Vec<RunRecord> = [("a", a), ("b", b)]
                .into_iter()
                .filter_map(|(id, h)| h.map(|h| run_with(id, h)))
                .collect();
            let full: Vec<(String, Vec<EpochPoint>)> = runs
                .iter()
                .map(|r| (r.id.clone(), r.progress.as_ref().unwrap().history.clone()))
                .collect();
            let ans = answer(
                runs,
                &[],
                true,
                Value::Null,
                Value::Null,
                Some(&parse_have(&have)),
            );
            if n == 3 || n == 6 {
                // Nothing changed: no row is sent again.
                for r in ans["runs"].as_array().unwrap() {
                    assert_eq!(r["progress"]["history"], serde_json::json!([]), "step {n}");
                }
            }
            let out = node_merge(&store, &ans);
            let merged = out["runs"].as_array().unwrap();
            assert_eq!(merged.len(), full.len(), "step {n}");
            for (m, (id, h)) in merged.iter().zip(&full) {
                assert_eq!(&m["id"], id, "step {n}");
                // As rows: JSON from node writes 10.0 as 10.
                let got: Vec<EpochPoint> =
                    serde_json::from_value(m["progress"]["history"].clone()).unwrap();
                assert_eq!(&got, h, "step {n}, run {id}");
                assert!(m["progress"].get("history_since").is_none());
            }
            store = out["store"].clone();
            have = out["have"].as_str().unwrap().to_string();
        }
        assert!(have.starts_with("a:1:"), "{have}");
    }

    #[test]
    fn logs_and_results_only_where_the_page_reads_them() {
        let mut done = run_with("d", vec![]);
        done.state = RunState::Done;
        let v = poll(vec![run_with("r", vec![]), done], "");
        assert_eq!(
            v["runs"][0]["log"].as_array().unwrap().len(),
            POLL_LOG_LINES
        );
        assert_eq!(v["runs"][0]["log"][0], "line 20");
        assert!(v["runs"][1].get("log").is_none());
        assert!(v["runs"][0]["progress"].get("history_keys").is_none());
        assert!(v.get("capability").is_none());
        assert!(v["runs"][0].get("run_dir").is_none() && v["runs"][0].get("output").is_none());
        // The catalog comes with a poll that names no history only.
        assert!(v.get("metrics").is_none());
        let first = answer(
            vec![],
            &[],
            false,
            Value::Null,
            Value::from("catalog"),
            None,
        );
        assert_eq!(first["metrics"], "catalog");
    }

    /// `TrainMetrics.constantOf` of `assets/train-metrics.js`: a learning rate that never changed
    /// is one line of text instead of a flat chart; any change, or fewer than 2 epochs, keeps it.
    #[test]
    fn a_constant_learning_rate_is_named_not_charted() {
        if !have_node() {
            return;
        }
        const DRIVER: &str = r#"
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const ctx = vm.createContext({ window: {}, console });
vm.runInContext(fs.readFileSync(process.env.BOP_TRAIN_METRICS, 'utf8'), ctx);
const C = ctx.window.TrainMetrics.constantOf, F = ctx.window.TrainMetrics.format;
const h = (lrs) => lrs.map((lr, i) => ({ epoch: i + 1, metrics: lr === null ? { map: 0.1 } : { lr, map: 0.1 } }));
// An f32 1e-4 widened to f64 is the same rate.
assert.strictEqual(C(h([1e-4, 0.0000999999974737875, 1e-4]), 'lr'), 1e-4);
assert.strictEqual(F('sci', C(h([1e-4, 1e-4]), 'lr')), '1.00e-4');
// Epochs without the key are left out; a gap is no change.
assert.strictEqual(C(h([5e-5, null, 5e-5]), 'lr'), 5e-5);
assert.strictEqual(C(h([1e-4, 9e-5]), 'lr'), null);
assert.strictEqual(C(h([1e-4]), 'lr'), null);
assert.strictEqual(C(h([1e-4, 1e-4]), 'missing'), null);
assert.strictEqual(C([], 'lr'), null);
"#;
        let js = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/train-metrics.js");
        let out = std::process::Command::new("node")
            .args(["-e", DRIVER])
            .env("BOP_TRAIN_METRICS", js)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "node: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
