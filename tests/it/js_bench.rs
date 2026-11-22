//! The Benchmark pages' pure rules, `assets/bench-core.js` (`BenchCore`), run in `node` without a
//! DOM: number formats, the goal catalog and its query (`?goal=f2&budget=500`, what the views and
//! `POST /v1/benchmark/goal` read), the row chips, the Compare columns' stored choice, the run's
//! per-combo states from the run poll, the CPU reference rule of the Run step, the run-time
//! estimate, old `/benchmark#...` addresses, the Images page's "disagrees with", the CSV, the
//! first-load text (twin of `report::first_load_text`), the frame dialog's label size and "latest
//! request wins".
//!
//! Then the pages themselves (bench-shell.js and the step scripts) on a stub DOM with a fake
//! clock and a fake `fetch` (`PAGE`): the run poll's period, out-of-order and aborted answers on
//! Compare, Tune, Images and History, History's delete question, and what a failed load leaves.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).
#![allow(clippy::unwrap_used)]

use serde_json::{Value, json};
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

/// `expr` evaluated in a context that loaded bench-core.js (`BC` = BenchCore), as JSON.
fn eval(expr: &str) -> Value {
    let core = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/bench-core.js");
    let driver = r#"
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({});
vm.runInContext(fs.readFileSync(process.env.BOP_BENCH_CORE_JS, 'utf8'), ctx);
vm.runInContext('var BC = BenchCore;', ctx);
const out = vm.runInContext('(' + process.env.BOP_EXPR + ')', ctx);
process.stdout.write(JSON.stringify(out === undefined ? null : out));
"#;
    let out = Command::new("node")
        .args(["-e", driver])
        .env("BOP_BENCH_CORE_JS", core)
        .env("BOP_EXPR", expr)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "node: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn formats() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"[BC.fmt.n2(0.8149), BC.fmt.n2(null), BC.fmt.n1(3.14), BC.fmt.ms(420.4), BC.fmt.ms(1400), BC.fmt.ms(12.34),
            BC.fmt.ms(4.56), BC.fmt.ms(undefined), BC.fmt.pct(0.873), BC.fmt.ci({value: 0.81, low: 0.79, high: 0.84}),
            BC.fmt.ci({value: 0.81, low: null, high: null}), BC.fmt.signed(0.031), BC.fmt.signed(-0.5, 1), BC.fmt.signed(0.0001),
            BC.fmt.duration(42), BC.fmt.duration(600), BC.fmt.duration(9000), BC.fmt.int(1234567)]"#,
    );
    assert_eq!(
        v,
        json!([
            "0.81",
            "-",
            "3.1",
            "420 ms",
            "1.4 s",
            "12 ms",
            "4.6 ms",
            "-",
            "87%",
            "±0.03",
            "",
            "+0.03",
            "−0.5",
            "±0.00",
            "42 s",
            "10 min",
            "2.5 h",
            "1,234,567"
        ])
    );
}

#[test]
fn goals_and_the_tried_goal_query() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"({
          parsed: ["f2", "precision:0.85", "recall_at_fa:2.5", "fpr", "grade", "nope", "precision:x"].map(BC.parseGoal),
          specs: [BC.goalSpec("recall_at_fa", 5), BC.goalSpec("precision", 0.900000001), BC.goalSpec("f1", 3), BC.goalSpec("recall", NaN), BC.goalSpec("x", 1)],
          problems: [BC.targetProblem("precision", 1.2), BC.targetProblem("precision", 0.9), BC.targetProblem("f1", null), BC.targetProblem("recall_at_fa", NaN)],
          kinds: BC.GOALS.map(function (g) { return g.kind + (g.more ? "+" : "") + (g.rankOnly ? "!" : ""); }),
          pairs: BC.goalPairs({optimizeFor: "f2", tuneFor: "f1", classes: ["person", "car"], budget: 500}),
          pairsAll: BC.goalPairs({optimizeFor: "ap50", tuneFor: "f05", classes: [], budget: null}),
          tried: BC.tryParams("?set=cameras&goal=recall_at_fa%3A5&budget=500&scene=day&classes=person%2Ccar&tune_for="),
          none: BC.tryParams(""),
          withTry: BC.withTry("/benchmark/tune?model=a&goal=f1#x", [["goal", "f2"], ["budget", "500"]]),
          withTryBare: BC.withTry("/benchmark/run", [["goal", "f2"]]),
          withNothing: BC.withTry("/benchmark/run", []),
          without: BC.withoutTry("/benchmark/compare?set=cameras&goal=f2&budget=none&scene=day#top"),
          withoutAll: BC.withoutTry("/benchmark/compare?goal=f2")
        })"#,
    );
    assert_eq!(
        v["parsed"],
        json!([
            {"kind": "f2", "target": null},
            {"kind": "precision", "target": 0.85},
            {"kind": "recall_at_fa", "target": 2.5},
            {"kind": "fpr", "target": 0.05},
            {"kind": "grade", "target": null},
            null,
            {"kind": "precision", "target": 0.9}
        ])
    );
    assert_eq!(
        v["specs"],
        json!(["recall_at_fa:5", "precision:0.9", "f1", "recall:0.9", null])
    );
    assert_eq!(v["problems"][1], Value::Null);
    assert_eq!(v["problems"][2], Value::Null);
    assert!(v["problems"][0].as_str().unwrap().contains("0.01 to 0.99"));
    assert!(
        v["problems"][3]
            .as_str()
            .unwrap()
            .contains("enter a number")
    );
    // Every goal the server knows, the rank-only ones under More.
    assert_eq!(
        v["kinds"],
        json!([
            "f1",
            "f2",
            "f05",
            "recall_at_fa",
            "precision",
            "recall",
            "youden+",
            "fpr+",
            "ap50+!",
            "ap50_95+!",
            "roc_auc+!",
            "grade+!"
        ])
    );
    for k in v["kinds"].as_array().unwrap() {
        let k = k.as_str().unwrap().trim_end_matches(['+', '!']);
        let spec = match k {
            "recall_at_fa" => "recall_at_fa:5".to_string(),
            "precision" | "recall" => format!("{k}:0.9"),
            "fpr" => "fpr:0.05".to_string(),
            _ => k.to_string(),
        };
        assert!(
            spec.parse::<blue_onyx_prism::benchmark::eval::Optimize>()
                .is_ok(),
            "{spec}: the server parses what the dialog sends"
        );
    }
    assert_eq!(
        v["pairs"],
        json!([
            ["goal", "f2"],
            ["tune_for", "f1"],
            ["classes", "person,car"],
            ["budget", "500"]
        ])
    );
    assert_eq!(
        v["pairsAll"],
        json!([
            ["goal", "ap50"],
            ["tune_for", "f05"],
            ["classes", "all"],
            ["budget", "none"]
        ])
    );
    assert_eq!(
        v["tried"],
        json!([
            ["goal", "recall_at_fa:5"],
            ["budget", "500"],
            ["classes", "person,car"]
        ])
    );
    assert_eq!(v["none"], json!([]));
    assert_eq!(v["withTry"], "/benchmark/tune?model=a&goal=f2&budget=500#x");
    assert_eq!(v["withTryBare"], "/benchmark/run?goal=f2");
    assert_eq!(v["withNothing"], "/benchmark/run");
    assert_eq!(v["without"], "/benchmark/compare?set=cameras&scene=day#top");
    assert_eq!(v["withoutAll"], "/benchmark/compare");
}

#[test]
fn row_chips_columns_and_texts() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"({
          chips: BC.rowChips({isDefault: true, flags: {trainedHere: true, trainedOnShare: 0.25, tooSlow: true, unmet: true, served: true, staleRules: true, lowConfidence: "only 3 camera days", relative: true, flat: true}}).map(function (c) { return [c.cls, c.text]; }),
          tip: BC.rowChips({flags: {trainedHere: true, trainedOnShare: 0.25}})[0].tip,
          lowTip: BC.rowChips({flags: {lowConfidence: "only 3 camera days"}})[0].tip,
          none: BC.rowChips({flags: {}}),
          cols: [BC.columnsOf('["p99","f1","bogus"]'), BC.columnsOf("not json"), BC.columnsOf(["burst", "cpu"]), BC.columnsOf(null)],
          values: ["f1", "ap50_95", "roc_auc", "p50", "p99", "burst", "first_load", "cpu"].map(function (k) {
            return BC.columnValue({f1: 0.7, ap5095: 0.41, rocAuc: 0.93, agreement: "match", speed: {p50Ms: 150, p99Ms: 1260, firstLoad: {compileMs: 2100, warmupMs: 500, cache: "hit"}}}, k);
          }),
          agreements: ["minor", "not_checked", null, "some_new_verdict"].map(BC.agreementText),
          loads: [[2100, "hit"], [24400, "miss"], [800, ""], [3000, "disabled"], [3000, "unknown"], [-5, "hit"]].map(function (a) { return BC.loadText(a[0], a[1]); }),
          firstLoad: [BC.firstLoadText(null), BC.firstLoadText({compileMs: 0, warmupMs: 0, cache: ""}), BC.firstLoadText({compileMs: 900, warmupMs: 100, cache: "miss"})]
        })"#,
    );
    assert_eq!(
        v["chips"],
        json!([
            ["tag", "default"],
            ["bn-chip warn", "trained here"],
            ["bn-chip bad", "too slow"],
            ["bn-chip bad", "target not met"],
            ["bn-chip same", "served"],
            ["bn-chip warn", "stale rules"],
            ["bn-chip warn", "low confidence"],
            ["bn-chip same", "relative"],
            ["bn-chip same", "flat curve"]
        ])
    );
    assert!(v["tip"].as_str().unwrap().contains("25% of these frames"));
    assert_eq!(v["lowTip"], "only 3 camera days");
    assert_eq!(v["none"], json!([]));
    assert_eq!(
        v["cols"],
        json!([["f1", "p99"], [], ["burst", "cpu"], []]),
        "known keys, in the table's order"
    );
    assert_eq!(
        v["values"],
        json!([
            "0.70",
            "0.41",
            "0.93",
            "150 ms",
            "1.3 s",
            "not run",
            "2.1 s, cached",
            "agrees with CPU"
        ])
    );
    assert_eq!(
        v["agreements"],
        json!([
            "minor differences",
            "not checked against CPU",
            "not checked against CPU",
            "some new verdict"
        ])
    );
    // Twin of the report's text.
    let loads = v["loads"].as_array().unwrap();
    for (i, (ms, cache)) in [
        (2100.0, "hit"),
        (24400.0, "miss"),
        (800.0, ""),
        (3000.0, "disabled"),
        (3000.0, "unknown"),
        (-5.0, "hit"),
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            loads[i],
            blue_onyx_prism::benchmark::report::first_load_text(*ms, cache),
            "{ms} {cache}"
        );
    }
    assert_eq!(v["firstLoad"], json!(["-", "-", "0.9 s, compiled"]));
}

#[test]
fn the_runs_combos_from_the_poll() {
    if !have_node() {
        return;
    }
    let run = json!({
        "running": true, "state": "running",
        "request": {"models": ["A", "b"], "devices": ["openvino:cpu", "openvino:gpu"]},
        "progress": {"stage": "benchmark", "model": "b", "device": "openvino:cpu", "phase": "timing", "done": 812, "total": 1420},
        "models": [{"model": "a", "devices": [
            {"device": "openvino:cpu", "run": {}},
            {"device": "openvino:gpu.0", "error": "compile failed"}
        ]}]
    });
    let v = eval(&format!(
        "BC.comboStates({run}).map(function (c) {{ return [c.model, c.device, c.state, c.text, c.now]; }})"
    ));
    assert_eq!(
        v,
        json!([
            ["a", "openvino:cpu", "done", "done", false],
            [
                "a",
                "openvino:gpu",
                "failed",
                "failed: compile failed",
                false
            ],
            [
                "b",
                "openvino:cpu",
                "timing",
                "scoring 812 of 1,420 images",
                true
            ],
            ["b", "openvino:gpu", "queued", "queued", false]
        ])
    );
    // Each model on its own device (no devices asked), compiling; a cancelled run's rest.
    let own = json!({
        "running": true, "request": {"models": ["m"], "devices": []},
        "progress": {"stage": "benchmark", "model": "m", "device": "ort:coreml", "phase": "loading", "done": 0, "total": 0},
        "models": []
    });
    let cancelled = json!({
        "running": false, "state": "cancelled", "request": {"models": ["m", "n"], "devices": ["ort:cpu"]},
        "progress": null, "models": [{"model": "m", "devices": [{"device": "ort:cpu", "run": {}}]}]
    });
    let v = eval(&format!(
        "[BC.comboStates({own}), BC.comboStates({cancelled}), BC.comboStates(null), BC.comboStates({{}})].map(function (l) {{ return l.map(function (c) {{ return [c.model, c.device, c.state, c.text]; }}); }})"
    ));
    assert_eq!(
        v,
        json!([
            [["m", "ort:coreml", "loading", "compiling"]],
            [
                ["m", "ort:cpu", "done", "done"],
                ["n", "ort:cpu", "skipped", "not run (cancelled)"]
            ],
            [],
            []
        ])
    );
}

#[test]
fn the_cpu_reference_and_the_estimate() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"(function () {
          var d = function (spec, checked, cpu, runnable) { return {spec: spec, checked: checked, cpu: cpu, runnable: runnable !== false}; };
          return {
            two: BC.cpuReference([d("openvino:gpu", true, false), d("ort:coreml", true, false), d("ort:cpu", false, true), d("openvino:cpu", false, true)], null),
            dismissed: BC.cpuReference([d("openvino:gpu", true, false), d("ort:coreml", true, false), d("openvino:cpu", false, true)], "openvino:cpu"),
            ortOnly: BC.cpuReference([d("ort:coreml", true, false), d("ort:directml", true, false), d("ort:cpu", false, true)], null),
            hasCpu: BC.cpuReference([d("openvino:gpu", true, false), d("openvino:cpu", true, true)], null),
            one: BC.cpuReference([d("openvino:gpu", true, false), d("openvino:cpu", false, true)], null),
            notRunnable: BC.cpuReference([d("a", true, false), d("b", true, false), d("openvino:cpu", false, true, false)], null),
            est: BC.estimate({models: ["m", "n"], devices: ["openvino:gpu", "openvino:cpu"], timing: {m: {"openvino:gpu.0": 100, "openvino:cpu": 400}, n: {}}, images: 50, repeat: 2, warmup: 3}),
            noTime: BC.estimate({models: ["m"], devices: ["ort:cpu"], timing: {m: {}}, images: 10, repeat: 1, warmup: 0}),
            noImages: BC.estimate({models: ["m"], devices: ["openvino:cpu"], timing: {m: {"openvino:cpu": 100}}, images: 0, repeat: 1, warmup: 0})
          };
        })()"#,
    );
    assert_eq!(v["two"], "openvino:cpu", "OpenVINO's CPU first");
    assert_eq!(
        v["dismissed"],
        Value::Null,
        "unticked by the user: never again"
    );
    assert_eq!(v["ortOnly"], "ort:cpu");
    assert_eq!(v["hasCpu"], Value::Null);
    assert_eq!(v["one"], Value::Null, "one device needs no reference");
    assert_eq!(v["notRunnable"], Value::Null);
    // m: gpu 2 s warm-up (3 x 0.1 s, at least 2) + 2 x 50 x 0.1 = 12 s; cpu min(20, 3 x 0.4 = 1.2
    // -> 2) + 2 x 50 x 0.4 = 42 s; n: unknown.
    assert_eq!(v["est"], json!({"seconds": 54, "known": 2, "pairs": 4}));
    assert_eq!(
        v["noTime"],
        json!({"seconds": null, "known": 0, "pairs": 1})
    );
    assert_eq!(v["noImages"]["seconds"], Value::Null);
}

/// The burst test's time per combo (`burstSeconds`, the twin of `src/benchmark/burst.rs`'s
/// schedule: 3 bursts, the next one K x I + 2 s after the one before and 2 s after its last
/// answer, the worker's load 3 s) and the Run estimate with it.
#[test]
fn the_burst_test_estimate() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"(function () {
          var b = {cameras: 8, images: 4, interval_ms: 500};
          return {
            fast: BC.burstSeconds(b, 100),
            unknown: BC.burstSeconds(b, null),
            off: BC.burstSeconds(null, 100),
            one: BC.burstSeconds({cameras: 1, images: 1, interval_ms: 50}, 20),
            est: BC.estimate({models: ["m", "n"], devices: ["openvino:gpu", "openvino:cpu"], timing: {m: {"openvino:gpu.0": 100, "openvino:cpu": 400}, n: {}}, images: 50, repeat: 2, warmup: 3, burst: b}),
            noBurst: BC.estimate({models: ["m"], devices: ["openvino:cpu"], timing: {m: {"openvino:cpu": 400}}, images: 50, repeat: 2, warmup: 3})
          };
        })()"#,
    );
    let f = |k: &str| v[k].as_f64().unwrap();
    // 100 ms: a burst lasts max(1.5 s of sends + 0.1, 32 x 0.1 s queued) = 3.2 s; the next starts
    // max(4 s, 3.2 + 2 s) later: 3 + 5.2 + 5.2 + 3.2.
    assert!((f("fast") - 16.6).abs() < 1e-9, "{v}");
    // Unknown speed: the sends alone, 3 + 4 + 4 + 1.5.
    assert!((f("unknown") - 12.5).abs() < 1e-9, "{v}");
    assert_eq!(f("off"), 0.0);
    // One request of 20 ms: a burst lasts 0.02 s, the next starts max(0.05 + 2, 0.02 + 2) s
    // later: 3 + 2.05 + 2.05 + 0.02.
    assert!((f("one") - 7.12).abs() < 1e-9, "{v}");
    // The run estimate: m on the GPU 12 s + 16.6 s, on the CPU 42 s + (3 + 14.8 + 14.8 +
    // 12.8) s; n unknown.
    let est = &v["est"];
    assert!(
        (est["seconds"].as_f64().unwrap() - (12.0 + 16.6 + 42.0 + 45.4)).abs() < 1e-9,
        "{est}"
    );
    assert!(
        (est["burst"].as_f64().unwrap() - (16.6 + 45.4)).abs() < 1e-9,
        "{est}"
    );
    assert_eq!(
        (est["known"].clone(), est["pairs"].clone()),
        (json!(2), json!(4))
    );
    assert_eq!(
        v["noBurst"]["burst"],
        Value::Null,
        "no burst asked: no share"
    );
    assert_eq!(v["noBurst"]["seconds"], json!(42));
}

/// A Compare row's burst texts (`burstFacts`) and its columns; the run poll's burst phase.
#[test]
fn the_burst_texts_and_the_burst_phase() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"(function () {
          var b = {cameras: 8, images: 4, intervalMs: 500, bursts: 3, requests: 96, answered: 94, p50Ms: 120, p95Ms: 480, p99Ms: 610,
            lastAnswerP50Ms: 900, lastAnswerP95Ms: 1600, refused: 2, refusedBy: "queue_full 2", maxQueueDepth: 12, queueCapacity: 15, inflight: 2, notes: []};
          var row = {speed: {burstP95Ms: 480, burst: b}}, none = {speed: {}}, failed = {speed: {burstP95Ms: null, burst: {cameras: 8, images: 4, intervalMs: 500, error: "its worker did not load", notes: []}}};
          var run = {running: true, request: {models: ["m"], devices: ["openvino:cpu"]}, models: [],
            progress: {stage: "benchmark", model: "m", device: "openvino:cpu", phase: "burst", done: 40, total: 96}};
          return {
            facts: BC.burstFacts(b), failed: BC.burstFacts(failed.speed.burst), none: BC.burstFacts(undefined),
            cols: [BC.columnValue(row, "burst"), BC.columnValue(row, "last_answer"), BC.columnValue(none, "burst"), BC.columnValue(none, "last_answer"), BC.columnValue(failed, "burst")],
            hasCol: BC.COLUMNS.some(function (c) { return c.key === "last_answer"; }),
            combo: BC.comboStates(run).map(function (c) { return [c.state, c.text]; })
          };
        })()"#,
    );
    assert_eq!(
        v["facts"],
        json!({
            "load": "8 cameras × 4 images, 500 ms apart",
            "short": "8 × 4 × 500 ms",
            "error": null,
            "latency": "120 ms · 480 ms · 610 ms",
            "lastAnswer": "900 ms · 1.6 s",
            "refused": "2 of 96 (queue_full 2)",
            "queue": "up to 12 waiting (queue 15, 2 in flight)"
        })
    );
    assert_eq!(v["failed"]["error"], "its worker did not load");
    assert_eq!(v["failed"]["latency"], "-");
    assert_eq!(v["none"], Value::Null);
    assert_eq!(
        v["cols"],
        json!(["480 ms", "1.6 s", "not run", "not run", "no answer"])
    );
    assert_eq!(v["hasCol"], true);
    assert_eq!(
        v["combo"],
        json!([["burst", "burst test: 40 of 96 answered"]])
    );
}

#[test]
fn old_addresses_map_to_the_new_pages() {
    if !have_node() {
        return;
    }
    let v = eval(
        r##"["#tab-compare", "#tab-run", "#tab-tune", "#tab-history", "#tab-details", "#tab-overview", "#tab-results",
            "#tab-profiles", "#tab-apply", "#tab-advanced", "#tab-confidence", "#tab-models", "#m-IPcam-general",
            "#m-a%20b", "#d-yolo26s", "#m-%E0%A4%A", "#nothing", "", null, "tab-run"].map(BC.mapHash)"##,
    );
    assert_eq!(
        v,
        json!([
            "/benchmark/compare",
            "/benchmark/run",
            "/benchmark/tune",
            "/benchmark/history",
            "/benchmark/compare",
            "/benchmark/tune",
            "/benchmark/tune",
            "/benchmark/tune",
            "/benchmark/tune",
            "/benchmark/compare",
            "/benchmark/tune",
            "/benchmark/compare",
            "/benchmark/tune?model=IPcam-general",
            "/benchmark/tune?model=a%20b",
            "/benchmark/compare?model=yolo26s",
            null,
            null,
            null,
            null,
            "/benchmark/run"
        ])
    );
}

#[test]
fn disagreements_and_csv() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"(function () {
          var f = function (file, missed, fp) { return {set: "s", file: file, missed: missed, falsePositives: fp}; };
          var a = [f("1", 1, 0), f("2", 0, 0), f("3", 0, 2), f("4", 1, 1), f("only-a", 1, 0)];
          var b = [f("1", 0, 0), f("2", 0, 0), f("3", 0, 1), f("4", 2, 0), f("only-b", 0, 1)];
          var view = {goal: {metric: "F2"}, rows: [
            {block: "ok", rank: 1, model: "m", device: "ort:cpu", metric: {value: 0.81, low: 0.79, high: 0.84}, recall: 0.9, precision: 0.7, f1: 0.79, f2: 0.85, f05: 0.73, fa100: 3.1, confidence: 0.34, ap50: 0.7, ap5095: 0.45, rocAuc: 0.93, speed: {p50Ms: 100, p95Ms: 150, p99Ms: 200}, flags: {lowConfidence: "few"}},
            {block: "too_slow", rank: 2, model: 'say "x", y', device: "openvino:cpu", metric: null, flags: {tooSlow: true}, speed: {}}
          ]};
          return {
            dis: BC.disagree(a, b).map(function (x) { return [x.file, x.other.missed]; }),
            empty: BC.disagree([], b).length,
            csv: BC.compareCsv(view),
            cell: ["heldout:cam:back", "heldout:camscene:ir:a", "camera:back", "heldout:all"].map(BC.cellTag)
          };
        })()"#,
    );
    assert_eq!(
        v["dis"],
        json!([["1", 0], ["4", 2]]),
        "1: only a misses; 2: both fine; 3: both alert; 4: both miss, only a alerts"
    );
    assert_eq!(
        v["dis"].as_array().unwrap().len(),
        2,
        "frames only one combo has are left out"
    );
    assert_eq!(v["empty"], 0);
    let csv = v["csv"].as_str().unwrap();
    let lines: Vec<&str> = csv.split("\r\n").collect();
    assert_eq!(
        lines[0],
        "rank,model,device,F2,low,high,recall,precision,f1,f2,f05,fa_per_100_empty,confidence,ap50,ap50_95,roc_auc,p50_ms,p95_ms,p99_ms,flags"
    );
    assert_eq!(
        lines[1],
        "1,m,ort:cpu,0.81,0.79,0.84,0.9,0.7,0.79,0.85,0.73,3.1,0.34,0.7,0.45,0.93,100,150,200,low confidence"
    );
    assert_eq!(
        lines[2], r#","say ""x"", y",openvino:cpu,,,,,,,,,,,,,,,,,too slow"#,
        "quoted, no rank outside the ranked block"
    );
    assert_eq!(lines[3], "", "ends with a line break");
    assert_eq!(v["cell"], json!([true, true, false, false]));
}

/// The scatter's numbers sit clear of each other and of every point, however close the points;
/// a one-value range is no range; a model whose training lists do not read says so.
#[test]
fn labels_ranges_and_unknown_training() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"(function () {
          var pts = [{x: 100, y: 100, text: "1"}, {x: 104, y: 101, text: "2"}, {x: 98, y: 96, text: "3"}, {x: 300, y: 50, text: "4"}];
          var spots = BC.placeLabels(pts);
          var boxes = spots.map(function (s, i) {
            var w = pts[i].text.length * 7, x1 = s.anchor === "middle" ? s.x - w / 2 : s.anchor === "start" ? s.x : s.x - w;
            return {x1: x1, x2: x1 + w, y1: s.y - 11, y2: s.y};
          });
          var clash = 0;
          boxes.forEach(function (b, i) {
            boxes.forEach(function (c, j) { if (i < j && b.x1 < c.x2 && b.x2 > c.x1 && b.y1 < c.y2 && b.y2 > c.y1) clash++; });
            pts.forEach(function (p) { if (b.x1 < p.x + 7 && b.x2 > p.x - 7 && b.y1 < p.y + 7 && b.y2 > p.y - 7) clash++; });
          });
          return {clash: clash, alone: spots[3], ranges: [BC.rangeText([0.2, 0.2]), BC.rangeText([0.2, 0.24]), BC.rangeText(null), BC.rangeText([0.201, 0.204])],
            unknown: BC.rowChips({flags: {trainedHere: true, unknownSeen: "its lists do not read"}}).map(function (c) { return c.text + ": " + c.tip; })};
        })()"#,
    );
    assert_eq!(v["clash"], 0, "{v}");
    assert_eq!(
        v["alone"],
        json!({"x": 300, "y": 40, "anchor": "middle"}),
        "a lone point's number goes right above it"
    );
    assert_eq!(v["ranges"], json!([null, "0.20–0.24", null, null]));
    let unknown = v["unknown"][0].as_str().unwrap();
    assert!(
        unknown.starts_with("training unknown: ") && unknown.contains("its lists do not read"),
        "{unknown}"
    );
}

/// The page harness of the tests below: a stub DOM (elements keep their children, attributes,
/// listeners), a fake clock behind `setTimeout`, and a fake `fetch` whose requests a test answers,
/// holds, aborts, or answers with headers first. live.js loads before `document` exists (only
/// PrismPoll); the charts are stubbed. `process.argv[2]` is the assets folder.
const PAGE: &str = r##"
const assert = require('assert'), fs = require('fs'), path = require('path'), vm = require('vm');
const ASSETS = process.argv[2];
const tick = () => new Promise(r => setImmediate(r));
const flush = async () => { for (let i = 0; i < 8; i++) await tick(); };

// A page of the Benchmark area on a stub DOM, a fake clock and a fake `fetch`. o: {path, search,
// step, hasResults, route(req) -> {json, status, delay} | {hold: true}}.
function page(o) {
  o = o || {};
  const W = {now: 0, seq: 0, timers: [], reqs: [], created: [], byId: {}, posted: []};
  class El {
    constructor(tag, id) {
      this.tagName = String(tag).toUpperCase(); this.id = id || ''; this.attrs = {}; this.children = [];
      this.listeners = {}; this.dataset = {}; this.style = {}; this._text = ''; this.hidden = false;
      this.className = ''; this.value = ''; this.checked = false; this.disabled = false; this.open = false;
      this.parentNode = null; this.isConnected = true; this.selected = false;
      const cls = new Set();
      this.classList = {add: (...c) => c.forEach(x => cls.add(x)), remove: (...c) => c.forEach(x => cls.delete(x)),
        contains: c => cls.has(c), toggle: (c, on) => { if (on === undefined ? !cls.has(c) : on) cls.add(c); else cls.delete(c); }};
      W.created.push(this);
    }
    get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
    set textContent(v) { this._text = String(v === null || v === undefined ? '' : v); this.children = []; }
    get firstChild() { return this.children[0] || null; }
    get clientWidth() { return 800; }
    get clientHeight() { return 450; }
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'hidden') this.hidden = true; if (k === 'id') this.id = String(v); if (k === 'class') this.className = String(v); }
    getAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }
    hasAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k); }
    removeAttribute(k) { delete this.attrs[k]; if (k === 'hidden') this.hidden = false; }
    appendChild(c) { if (c && c.parentNode) c.remove(); this.children.push(c); c.parentNode = this; return c; }
    append(...cs) { cs.forEach(c => this.appendChild(typeof c === 'object' ? c : W.text(c))); }
    replaceChildren(...cs) { this.children.forEach(c => { c.parentNode = null; }); this.children = []; this._text = ''; cs.forEach(c => this.appendChild(typeof c === 'object' ? c : W.text(c))); }
    remove() { if (this.parentNode) { this.parentNode.children = this.parentNode.children.filter(x => x !== this); this.parentNode = null; } }
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
    removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter(x => x !== f); }
    dispatch(t, extra) {
      let stop = false;
      const ev = Object.assign({type: t, target: this, currentTarget: this, defaultPrevented: false,
        preventDefault() { ev.defaultPrevented = true; }, stopPropagation() {}, stopImmediatePropagation() { stop = true; }}, extra || {});
      for (const f of (this.listeners[t] || []).slice()) { if (stop) break; f(ev); }
      if (!stop && typeof this['on' + t] === 'function') this['on' + t](ev);
      return ev;
    }
    // A form's submit: the fields it would post (hidden inputs and ticked boxes) unless a
    // listener prevents it; kept in W.posted.
    requestSubmit(by) { const ev = this.dispatch('submit', {submitter: by || null}); if (!ev.defaultPrevented) W.posted.push(this.fields()); }
    submit() { W.posted.push(this.fields()); }
    fields() { return this.all().filter(e => e.tagName === 'INPUT' && e.name && (e.type !== 'checkbox' || e.checked)).map(e => [e.name, e.value]); }
    insertBefore(c, ref) { if (c.parentNode) c.remove(); const i = this.children.indexOf(ref); this.children.splice(i < 0 ? this.children.length : i, 0, c); c.parentNode = this; return c; }
    click() { if (!this.disabled) this.dispatch('click'); }
    // A select (or input) changed by the user.
    choose(v) { this.value = v; this.dispatch('change'); this.dispatch('input'); }
    focus() {} blur() {} scrollIntoView() {} contains() { return false; }
    showModal() { this.open = true; } close() { this.open = false; this.dispatch('close'); }
    // Simple selectors only (`tag`, `.cls`, `tag.cls`); anything else finds nothing.
    querySelectorAll(sel) {
      const m = /^([a-z]*)(?:\.([\w-]+))?$/i.exec(String(sel));
      if (!m || (!m[1] && !m[2])) return [];
      return this.all().filter(e => (!m[1] || e.tagName === m[1].toUpperCase()) && (!m[2] || (' ' + (e.getAttribute('class') || e.className) + ' ').indexOf(' ' + m[2] + ' ') >= 0));
    }
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
    closest() { return null; }
    getBoundingClientRect() { return {left: 0, top: 0, width: 800, height: 450, right: 800, bottom: 450}; }
    // Every element under this one (depth first); whether one has class `cls`.
    all() { const out = []; const walk = e => { e.children.forEach(c => { out.push(c); walk(c); }); }; walk(this); return out; }
    has(cls) { return this.all().some(e => (' ' + e.className + ' ').indexOf(' ' + cls + ' ') >= 0); }
  }
  W.El = El;
  W.text = t => { const n = new El('#text'); n._text = String(t); return n; };
  W.$ = id => { if (!W.byId[id]) W.byId[id] = new El('div', id); return W.byId[id]; };
  // o.strict: only the elements a test made (W.$) exist, as on a page without the others.
  const byId = id => o.strict ? (W.byId[id] || null) : W.$(id);
  // Skeletons the templates ship (a test sets the ones its page has).
  W.skel = (id, tall) => { const s = new El('span'); s.className = 'bn-skel' + (tall ? ' tall' : ''); W.$(id).replaceChildren(s); };
  const bench = W.$('bench'); bench.dataset.step = o.step || 'compare'; bench.dataset.hasResults = o.hasResults === false ? 'false' : 'true';
  const doc = {
    hidden: false, readyState: 'complete', cookie: '', body: new El('body'),
    getElementById: byId, querySelector: s => (o.qs ? o.qs(s, W) : null), querySelectorAll: s => (o.qsa ? o.qsa(s, W) : []),
    createElement: t => new El(t), createElementNS: (ns, t) => new El(t), createTextNode: W.text,
    createDocumentFragment: () => new El('#fragment'), addEventListener() {}, removeEventListener() {}
  };
  W.at = (ms, f) => { const t = {id: ++W.seq, at: W.now + (ms || 0), f}; W.timers.push(t); return t.id; };
  const ctx = vm.createContext({console, URL, URLSearchParams, AbortController});
  ctx.setTimeout = (f, ms) => W.at(ms, f);
  ctx.clearTimeout = id => { W.timers = W.timers.filter(t => t.id !== id); };
  W.load = name => vm.runInContext(fs.readFileSync(path.join(ASSETS, name), 'utf8'), ctx, {filename: name});
  W.load('live.js'); // before `document`: PrismPoll only
  const p = o.path || '/benchmark/compare';
  Object.assign(ctx, {
    window: ctx, document: doc, history: {replaceState() {}},
    location: {hash: '', search: o.search || '', pathname: p, href: 'https://h' + p + (o.search || ''), origin: 'https://h', replace() {}, assign() {}, reload() {}},
    localStorage: {getItem: () => null, setItem() {}}, matchMedia: () => ({matches: false}),
    addEventListener() {}, scrollTo() {}, CSS: {escape: s => String(s)}, Blob: function () {}
  });
  // fetch: each request is kept in W.reqs; `route` answers it (now or after `delay` ms of the fake
  // clock) or leaves it held: req.respond(json, status) answers it, req.headers(status) sends
  // only the headers (then req.bad() is a body that is not JSON). An abort rejects with an
  // AbortError, the body's json() too once the headers are in.
  ctx.fetch = (url, init) => {
    init = init || {};
    const req = {url: String(url), method: init.method || 'GET', sent: init.body ? String(init.body) : '', at: W.now, done: false, aborted: false};
    W.reqs.push(req);
    return new Promise((res, rej) => {
      let inBody = null;
      const abortError = () => { const e = new Error('The operation was aborted.'); e.name = 'AbortError'; return e; };
      const abort = () => {
        if (req.done) return;
        req.done = true; req.aborted = true;
        if (inBody) inBody.rej(abortError()); else rej(abortError());
      };
      if (init.signal) { if (init.signal.aborted) { abort(); return; } init.signal.addEventListener('abort', abort); }
      const reply = status => ({ok: status < 400, status, json: () => new Promise((a, b) => { inBody = {res: a, rej: b}; if (req.json !== undefined) a(JSON.parse(JSON.stringify(req.json))); })});
      req.respond = (json, status) => { if (req.done) return; req.done = true; req.json = json; res(reply(status || 200)); };
      req.headers = status => { if (req.done) return; res(reply(status || 200)); };
      req.bad = () => { if (req.done) return; req.done = true; const e = new SyntaxError('Unexpected token < in JSON'); if (inBody) inBody.rej(e); else req.json = undefined; };
      const r = o.route ? o.route(req, W) : null;
      if (!r || r.hold) return;
      const go = () => req.respond(r.json, r.status);
      if (r.delay) W.at(r.delay, go); else go();
    });
  };
  W.ctx = ctx;
  W.advance = async ms => {
    const end = W.now + ms;
    for (let guard = 0; guard < 100000; guard++) {
      const next = W.timers.filter(t => t.at <= end).sort((a, b) => a.at - b.at || a.id - b.id)[0];
      if (!next) break;
      W.timers = W.timers.filter(t => t !== next); W.now = next.at; next.f(); await flush();
    }
    W.now = end; await flush();
  };
  return W;
}
// The run poll's answer.
function runAnswer(running) {
  return {success: true, viewTag: 'v1', modelsTag: 'm1', running, state: running ? 'running' : 'done', run: {models: [], progress: running ? {stage: 'benchmark', percent: 10} : null}};
}
// deepStrictEqual across the vm's realm (its objects have the vm's prototypes).
const same = (a, b, msg) => assert.deepStrictEqual(JSON.parse(JSON.stringify(a === undefined ? null : a)), b, msg);
// A Benchmark page: bench-core.js, a stub of the charts (bench-charts.js draws SVG; its own tests
// are js_charts.rs), bench-shell.js, then `scripts`. The run poll answers "no run".
function benchPage(o, scripts) {
  const route = o.route;
  const W = page(Object.assign({}, o, {route: (req, w) => /only=run/.test(req.url) ? {json: runAnswer(false)} : route(req, w)}));
  W.load('bench-core.js');
  const chart = () => { const s = new W.El('svg'); s.setCursor = () => {}; return s; };
  W.ctx.PrismBench.charts = {lineChart: chart, scatter: chart, niceTicks: () => [0, 1], niceMax: v => v};
  W.load('bench-shell.js');
  scripts.forEach(W.load);
  return W;
}
const q = (req, k) => new URL(req.url, 'https://h').searchParams.get(k);
"##;

/// Run `PAGE` + `driver` in node (it asserts and exits non-zero on a failure).
fn run_page(driver: &str) {
    let dir = std::env::temp_dir().join(format!("bop-js-bench-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("driver.js");
    std::fs::write(&file, format!("{PAGE}\n{driver}\n")).unwrap();
    let out = Command::new("node")
        .arg(&file)
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets"))
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

/// F1: while a run goes the shell polls `?only=run` once per 1.5 s period (it kicked the fast loop
/// from inside its own poll, so polls went back to back: 61 in 3 s here), every 20 s once it
/// ended, and the fast loop takes over a period after the slow one sees a run start.
#[test]
fn the_run_poll_keeps_its_period() {
    if !have_node() {
        return;
    }
    run_page(
        r##"
(async () => {
  // A run goes: the run poll asks once per 1.5 s period (each answer takes 50 ms), never back to back.
  const S = {running: true};
  const W = page({path: '/benchmark/run', step: 'run', route: req => /only=run/.test(req.url) ? {json: runAnswer(S.running), delay: 50} : {json: {success: false}, status: 404}});
  W.load('bench-core.js'); W.load('bench-shell.js');
  await flush();
  await W.advance(3000);
  let at = W.reqs.filter(r => /only=run/.test(r.url)).map(r => r.at);
  assert.ok(at.length >= 2 && at.length <= 3, 'polls in 3 s of a run: ' + JSON.stringify(at));
  for (let i = 1; i < at.length; i++) assert.ok(at[i] - at[i - 1] >= 1500, 'a period between polls: ' + JSON.stringify(at));
  // The run ends: the fast loop parks, the slow one asks every 20 s.
  S.running = false;
  await W.advance(2000);
  const ended = W.reqs.length;
  await W.advance(19000);
  at = W.reqs.slice(ended).map(r => r.at);
  assert.strictEqual(at.length, 1, 'one poll per 20 s once the run ended: ' + JSON.stringify(at));
  // A run starts while the slow loop polls: the fast loop takes over at its period.
  S.running = true;
  const before = W.reqs.length;
  await W.advance(25000);
  at = W.reqs.slice(before).map(r => r.at);
  for (let i = 1; i < at.length; i++) assert.ok(at[i] - at[i - 1] >= 1500, 'a period between polls after a start: ' + JSON.stringify(at));
  assert.ok(at.length >= 2, 'the fast loop runs: ' + JSON.stringify(at));
  // BenchShell.kick() (a run started from the page) asks at once, then at the period.
  const k0 = W.reqs.length;
  W.ctx.BenchShell.kick(); await flush();
  assert.strictEqual(W.reqs.length, k0 + 1, 'kick asks at once');
  await W.advance(1400);
  assert.ok(W.reqs.length <= k0 + 2, 'then waits: ' + JSON.stringify(W.reqs.slice(k0).map(r => r.at)));
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##,
    );
}

/// F4: `PrismBench.getJSON` answers `{aborted: true}` for an abort while the body is read too (it
/// was a fake "The server answered HTTP 200." error), and a body that is not JSON is an error.
#[test]
fn an_abort_while_the_body_reads_is_an_abort() {
    if !have_node() {
        return;
    }
    run_page(
        r##"
(async () => {
  // PB.getJSON: an abort while the body is read is an abort, like one before the headers.
  const W = page({route: () => ({hold: true})});
  W.load('bench-core.js');
  const PB = W.ctx.PrismBench, out = {};
  const c1 = new AbortController();
  PB.getJSON('/v1/a', {signal: c1.signal}).then(j => { out.body = j; });
  await flush(); W.reqs[0].headers(200); await flush();
  c1.abort(); await flush();
  const c2 = new AbortController();
  PB.getJSON('/v1/b', {signal: c2.signal}).then(j => { out.head = j; });
  await flush(); c2.abort(); await flush();
  PB.getJSON('/v1/c').then(j => { out.bad = j; });
  await flush(); W.reqs[2].headers(502); await flush(); W.reqs[2].bad(); await flush();
  assert.strictEqual(out.body && out.body.aborted, true, 'aborted while reading the body: ' + JSON.stringify(out.body));
  assert.strictEqual(out.head && out.head.aborted, true, 'aborted before the headers: ' + JSON.stringify(out.head));
  same(out.bad, {success: false, _status: 502, message: 'The server answered HTTP 502 (see the Logs page).'}, 'a body that is not JSON');
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##,
    );
}

/// F4, F5, F7 on Compare: a failed first load replaces the skeletons with an error state; the set
/// before, aborted while its body reads, is no error; the latest set shows and clears the error.
#[test]
fn compare_latest_set_wins_and_a_failed_load_says_so() {
    if !have_node() {
        return;
    }
    run_page(
        r##"
(async () => {
  const view = (set, extra) => Object.assign({goal: {metric: 'F1', optimizeFor: 'f1', summary: 'F1'}, headline: 'view ' + set, notes: [], missing: [],
    sets: ['a', 'b', 'c'].map(id => ({id, title: id.toUpperCase(), frames: 10})), set: {id: set, scenes: []},
    rows: [{model: 'm1', device: 'ort:cpu', block: 'ok', rank: 1, position: 1, metric: {value: 0.8, low: 0.7, high: 0.9}, speed: {p95Ms: 100}, flags: {}}]}, extra || {});
  const W = benchPage({path: '/benchmark/compare', step: 'compare', route: () => ({hold: true})}, []);
  W.skel('c-headline'); W.skel('c-table', true);
  W.load('bench-compare.js');
  await flush();
  const loads = () => W.reqs.filter(r => /\/v1\/benchmark\/compare/.test(r.url));
  // F7: the first load fails: an error state, no skeletons left.
  loads()[0].respond({success: false, message: 'The results file is damaged.'}, 500); await flush();
  assert.ok(/damaged/.test(W.$('c-msg').textContent), 'the error shows: ' + W.$('c-msg').textContent);
  assert.ok(!W.$('c-headline').has('bn-skel') && !W.$('c-table').has('bn-skel'), 'no skeleton after a failed load');
  assert.ok(/not load/i.test(W.$('c-headline').textContent), 'the headline says it did not load: ' + W.$('c-headline').textContent);
  // F4 + F5: set b (its body still being read), then set c: b's abort is no error, c's answer shows.
  W.$('c-set').choose('b'); await flush();
  const b = loads()[1]; assert.strictEqual(q(b, 'set'), 'b');
  b.headers(200); await flush();
  W.$('c-set').choose('c'); await flush();
  const c = loads()[2]; assert.strictEqual(q(c, 'set'), 'c');
  assert.ok(b.aborted, 'the set before is aborted');
  assert.ok(!/HTTP 200/.test(W.$('c-msg').textContent), 'an abort is no error: ' + W.$('c-msg').textContent);
  assert.strictEqual(W.$('cmp-body').getAttribute('aria-busy'), 'true', 'still busy with c');
  c.respond({success: true, compare: view('c')}); await flush();
  assert.strictEqual(W.$('c-msg').textContent, '', 'a later success clears the load error');
  assert.ok(/view c/.test(W.$('c-headline').textContent), 'c shows');
  assert.strictEqual(W.$('cmp-body').getAttribute('aria-busy'), null);
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##,
    );
}

/// F5, F7 on Tune & save: a slower Compare answer of the set before never replaces the table, so
/// Save all posts the shown set's values; Save all waits while a set's table loads; a tune answer
/// of the set before does not pull the page back; a failed load leaves no skeleton.
#[test]
fn tune_saves_only_the_set_it_shows() {
    if !have_node() {
        return;
    }
    run_page(
        r##"
(async () => {
  const goal = {metric: 'F1', optimizeFor: 'f1', summary: 'F1'};
  const cmp = (set, conf) => ({success: true, compare: {goal, sets: ['a', 'b', 'c', 'd', 'e'].map(id => ({id, title: id, frames: 5})), set: {id: set},
    rows: [{model: 'm1', device: 'ort:cpu', block: 'ok', position: 1, confidence: conf, flags: {}, atSaved: {}, atSuggested: {}}]}});
  const grid = [0.1, 0.11, 0.33, 0.5].map(t => ({threshold: t, f1: 0.5, recall: 0.5, precision: 0.5, fa100: 1, tp: 1, fp: 1, fn: 1}));
  const tune = set => ({success: true, tune: {model: 'm1', device: 'ort:cpu', set, sets: ['a', 'b', 'c', 'd', 'e'], devices: ['ort:cpu'], grid, suggested: {threshold: 0.5}, saved: null, savable: true, objective: 'f1', objectiveName: 'F1', goal}});
  const W = benchPage({path: '/benchmark/tune', step: 'tune', route: req => {
    if (/tune\/images/.test(req.url)) return {json: {success: true, images: {total: 0, frames: []}}};
    if (/apply-profiles/.test(req.url)) return {json: {success: true, message: 'Saved.'}};
    return {hold: true};
  }}, ['bench-frames.js', 'bench-tune.js']);
  W.skel('t-chart', true); W.skel('t-side'); W.skel('t-examples', true); W.skel('t-all');
  await flush();
  const cmps = () => W.reqs.filter(r => /\/v1\/benchmark\/compare/.test(r.url));
  const tunes = () => W.reqs.filter(r => /\/v1\/benchmark\/tune\?/.test(r.url));
  const posts = () => W.reqs.filter(r => /apply-profiles/.test(r.url));
  cmps()[0].respond(cmp('a', 0.5)); await flush();
  tunes()[0].respond(tune('a')); await flush();
  // Set b, then c; c answers first, then b (slower): the table and Save all are c's.
  W.$('t-set').choose('b'); await flush(); const b = cmps()[1];
  W.$('t-set').choose('c'); await flush(); const c = cmps()[2];
  c.respond(cmp('c', 0.33)); await flush();
  b.respond(cmp('b', 0.11)); await flush();
  tunes().filter(r => !r.done).forEach(r => r.respond(tune(q(r, 'set')))); await flush();
  assert.strictEqual(q(tunes()[tunes().length - 1], 'set'), 'c', 'the tune view is c');
  W.$('t-save-all').click(); await flush();
  assert.strictEqual(posts().length, 1);
  assert.ok(/0\.33/.test(posts()[0].sent) && !/0\.11/.test(posts()[0].sent), 'Save all posts c\'s values: ' + posts()[0].sent);
  W.reqs.filter(r => !r.done && /compare|tune\?/.test(r.url)).forEach(r => r.respond(/compare/.test(r.url) ? cmp(q(r, 'set') || 'c', 0.33) : tune(q(r, 'set') || 'c'))); await flush();
  // Set d while its table loads: Save all refuses (the table is still c's).
  W.$('t-set').choose('d'); await flush();
  const nPosts = posts().length;
  W.$('t-save-all').click(); await flush();
  assert.strictEqual(posts().length, nPosts, 'nothing posted while the set\'s table loads');
  assert.ok(/load/i.test(W.$('t-result').textContent), 'says why: ' + W.$('t-result').textContent);
  cmps()[cmps().length - 1].respond(cmp('d', 0.44)); await flush();
  tunes().filter(r => !r.done).forEach(r => r.respond(tune(q(r, 'set')))); await flush();
  // A model change in flight, then set e: the old tune answer (set d) does not pull the page back.
  W.$('t-model').choose('m1'); await flush(); const oldTune = tunes()[tunes().length - 1];
  W.$('t-set').choose('e'); await flush();
  oldTune.respond(tune('d')); await flush();
  cmps()[cmps().length - 1].respond(cmp('e', 0.55)); await flush();
  assert.strictEqual(q(tunes()[tunes().length - 1], 'set'), 'e', 'the tune after set e asks for e');
  { // F7: the first load fails: the error, and no skeleton left.
    const V = benchPage({path: '/benchmark/tune', step: 'tune', route: req => /\/v1\/benchmark\/compare/.test(req.url) ? {json: {success: false, message: 'The results file is damaged.'}, status: 500} : {hold: true}}, ['bench-frames.js', 'bench-tune.js']);
    V.skel('t-chart', true); V.skel('t-side'); V.skel('t-examples', true); V.skel('t-all');
    await flush();
    assert.ok(/damaged/.test(V.$('t-error').textContent) && !V.$('t-error').hidden, 'the error shows');
    ['t-chart', 't-side', 't-examples', 't-all'].forEach(id => assert.ok(!V.$(id).has('bn-skel'), id + ': no skeleton after a failed load'));
    assert.ok(/not loaded/i.test(V.$('t-side').textContent));
  }
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##,
    );
}

/// F4, F5, F7 on Images: frames of the set before never show after a set change; an abort is no
/// error; a later success clears the error; a failed "disagrees with" leaves no skeleton.
#[test]
fn images_latest_choice_wins() {
    if !have_node() {
        return;
    }
    run_page(
        r##"
(async () => {
  const goal = {metric: 'F1', optimizeFor: 'f1', summary: 'F1'};
  const cmp = set => ({success: true, compare: {goal, sets: ['a', 'b'].map(id => ({id, title: id, frames: 5})), set: {id: set},
    rows: [{model: 'm1', device: 'ort:cpu', confidence: 0.5}, {model: 'm2', device: 'ort:cpu', confidence: 0.4}]}});
  const frames = (set, names) => ({success: true, images: {total: names.length, cameras: [], frames: names.map(n => ({set, file: n + '.jpg', width: 100, height: 50, gt: [], preds: []}))}});
  const imgs = W => W.reqs.filter(r => /tune\/images/.test(r.url));
  const cmps = W => W.reqs.filter(r => /\/v1\/benchmark\/compare/.test(r.url));
  const shown = W => W.$('i-grid').all().filter(e => e.tagName === 'BUTTON').map(e => e.textContent);
  {
    const W = benchPage({path: '/benchmark/images', step: 'images', route: () => ({hold: true})}, ['bench-frames.js', 'bench-images.js']);
    await flush();
    cmps(W)[0].respond(cmp('a')); await flush();
    const a = imgs(W)[0]; assert.strictEqual(q(a, 'set'), 'a');
    // Set b while a's frames load: a's answer never shows.
    W.$('i-set').choose('b'); await flush();
    a.respond(frames('a', ['a1'])); await flush();
    assert.ok(shown(W).indexOf('a1.jpg') < 0, 'set a\'s late frames never show under set b: ' + JSON.stringify(shown(W)));
    cmps(W)[1].respond(cmp('b')); await flush();
    const b = imgs(W)[imgs(W).length - 1]; assert.strictEqual(q(b, 'set'), 'b');
    b.respond(frames('b', ['b1', 'b2'])); await flush();
    same(shown(W), ['b1.jpg', 'b2.jpg'], 'b\'s frames only');
    // A kind whose body is still read, replaced by another: no error; a failure, then a success: the error goes.
    W.$('i-kind').choose('fa'); await flush(); const fa = imgs(W)[imgs(W).length - 1]; fa.headers(200); await flush();
    W.$('i-kind').choose('all'); await flush(); const all = imgs(W)[imgs(W).length - 1];
    assert.ok(W.$('i-error').hidden, 'an abort is no error: ' + W.$('i-error').textContent);
    all.respond({success: false, message: 'Disk error.'}, 500); await flush();
    assert.ok(!W.$('i-error').hidden && /Disk error/.test(W.$('i-error').textContent));
    W.$('i-scene').choose('day'); await flush();
    imgs(W)[imgs(W).length - 1].respond(frames('b', ['b3'])); await flush();
    assert.ok(W.$('i-error').hidden, 'a later success clears the error');
    same(shown(W), ['b3.jpg']);
  }
  { // F7: "disagrees with" fails on the first load: the error, no skeleton.
    const W = benchPage({path: '/benchmark/images', search: '?kind=dis', step: 'images', route: req => /tune\/images/.test(req.url) ? {json: {success: false, message: 'Disk error.'}, status: 500} : {hold: true}}, ['bench-frames.js', 'bench-images.js']);
    W.$('i-kind').value = 'dis';
    await flush();
    cmps(W)[0].respond(cmp('a')); await flush();
    assert.ok(/Disk error/.test(W.$('i-error').textContent), 'the error shows');
    assert.ok(!W.$('i-grid').has('bn-skel'), 'no skeleton after a failed load');
  }
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##,
    );
}

/// F2, F5, F7 on History: the delete question goes when the model or A / B changes and its yes
/// deletes the run it asked about (never another model's run of the same id); the latest A / B
/// choice's comparison shows; a failed load leaves no skeleton and a later success clears it.
#[test]
fn history_delete_asks_about_one_run() {
    if !have_node() {
        return;
    }
    run_page(
        r##"
(async () => {
  const run = (id, ts) => ({run_id: id, timestamp: ts, devices: [{device: 'ort:cpu', total_p95_ms: 100}], settings: {datasets: ['coco'], images: 10}});
  const hist = {success: true, models: {A: [run('r2', '2026-11-02T10:00:00Z'), run('r1', '2026-11-01T10:00:00Z')], B: [run('r2', '2026-11-03T10:00:00Z'), run('r0', '2026-10-01T10:00:00Z')]}, comparability: {}};
  const changes = s => ({success: true, changes: {summary: s, comparable: true, metrics: [{metric: 'f1', device: 'ort:cpu', before: 0.5, after: 0.6, verdict: 'better', why: s}]}});
  const W = benchPage({path: '/benchmark/history', search: '?model=A', step: 'history', route: req => {
    if (/history\/compare/.test(req.url)) return {hold: true};
    if (/history\/delete/.test(req.url)) return {json: {success: true, message: 'Deleted.'}};
    if (/\/v1\/benchmark\/history/.test(req.url)) return {json: hist};
    return {hold: true};
  }}, ['bench-history.js']);
  await flush();
  const deletes = () => W.reqs.filter(r => /history\/delete/.test(r.url));
  const delButton = ts => W.created.filter(e => e.tagName === 'BUTTON' && e.textContent === 'Delete' && e.isConnected && (e.getAttribute('aria-label') || '').indexOf(ts) >= 0).pop();
  // F2: Delete A's r2, then switch to model B before answering: the box goes, and its yes still
  // means A's r2 (never B's run with the same id).
  delButton('2026-11-02').click();
  assert.strictEqual(W.$('h-confirm').hidden, false, 'asked');
  W.$('h-model').choose('B'); await flush();
  assert.strictEqual(W.$('h-confirm').hidden, true, 'the question goes when the model changes');
  W.$('h-confirm-yes').onclick && W.$('h-confirm-yes').onclick(); await flush();
  assert.ok(deletes().every(r => /model=A/.test(r.sent) && /run_id=r2/.test(r.sent)), 'only ever A\'s r2: ' + JSON.stringify(deletes().map(r => r.sent)));
  // The A / B change hides it too.
  W.$('h-model').choose('A'); await flush();
  delButton('2026-11-02').click();
  W.$('h-a').choose('r1'); await flush();
  assert.strictEqual(W.$('h-confirm').hidden, true, 'the question goes when A changes');
  // F5: two A / B choices in a row; the first answers last: the table is the second's.
  const cmpReqs = () => W.reqs.filter(r => /history\/compare/.test(r.url));
  W.$('h-a').choose('r2'); await flush(); const first = cmpReqs()[cmpReqs().length - 1];
  W.$('h-b').choose('r2'); W.$('h-b').choose('r1'); await flush(); const second = cmpReqs()[cmpReqs().length - 1];
  assert.notStrictEqual(first, second);
  second.respond(changes('second')); await flush();
  first.respond(changes('first')); await flush();
  assert.ok(/second/.test(W.$('h-ab-summary').textContent), 'the latest choice\'s answer: ' + W.$('h-ab-summary').textContent);
  { // F7: a failed load: the error, no skeleton; a later success hides the error.
    let fail = true;
    const V = benchPage({path: '/benchmark/history', step: 'history', route: req => /\/v1\/benchmark\/history$/.test(req.url) ? (fail ? {json: {success: false, message: 'Damaged history.'}, status: 500} : {json: hist}) : {hold: true}}, []);
    V.skel('h-runs');
    V.load('bench-history.js'); await flush();
    assert.ok(!V.$('h-error').hidden && /Damaged/.test(V.$('h-error').textContent), 'the error shows');
    assert.ok(!V.$('h-runs').has('bn-skel'), 'no skeleton after a failed load');
    fail = false; V.ctx.BenchShell.viewChanged('run'); await flush();
    assert.ok(V.$('h-error').hidden, 'a later success hides the error');
  }
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##,
    );
}

/// "Latest request wins": a newer begin() aborts the one before, whose answer (and any aborted
/// one, or a list with one) never wins; the winner ends the ticket; no AbortController still goes
/// by the order; stop() aborts the one in flight.
#[test]
fn the_latest_request_wins() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"(function () {
          var made = [];
          var mk = function () { var c = {aborted: false, signal: {n: made.length}, abort: function () { c.aborted = true; }}; made.push(c); return c; };
          var L = BC.latest(mk), a = L.begin(), b = L.begin(), out = {};
          out.replaced = [made[0].aborted, made[1].aborted, a.current(), b.current(), b.init.signal === made[1].signal];
          out.wins = [a.wins({success: true}), L.busy(), b.wins({success: true}), L.busy(), b.wins({success: true})];
          var c = L.begin(); out.aborted = [c.wins({success: false, aborted: true}), L.busy()];
          var d = L.begin(); out.lists = [d.wins([{success: true}, {aborted: true}])];
          var e = L.begin(); out.lists.push(e.wins([{success: true}, {success: false}]));
          var z = L.begin(); L.stop(); out.stopped = [made[made.length - 1].aborted, z.wins({success: true}), L.busy()];
          var N = BC.latest(function () { return null; }), x = N.begin(), y = N.begin();
          out.plain = [JSON.stringify(x.init), x.wins({success: true}), y.wins({success: true})];
          out.noController = JSON.stringify(BC.latest().begin().init);
          return out;
        })()"#,
    );
    assert_eq!(v["replaced"], json!([true, false, false, true, true]));
    assert_eq!(
        v["wins"],
        json!([false, true, true, false, false]),
        "the replaced one never wins; the winner ends its ticket"
    );
    assert_eq!(
        v["aborted"],
        json!([false, false]),
        "an aborted answer ends it, unshown"
    );
    assert_eq!(
        v["lists"],
        json!([false, true]),
        "a list wins only with no abort in it"
    );
    assert_eq!(v["stopped"], json!([true, false, false]));
    assert_eq!(
        v["plain"],
        json!(["{}", false, true]),
        "without a controller the order decides"
    );
    assert_eq!(
        v["noController"], "{}",
        "no AbortController here (node's vm context)"
    );
}

/// The frame dialog's labels render at 13 CSS px whatever the frame's size and the screen's:
/// the size is in image pixels for the fitted scale ("meet"), the halo 3 px and the gap 4 px;
/// unmeasured, about 1/32 of the shorter side (at least 12).
#[test]
fn the_frame_labels_keep_their_size_on_screen() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"[[1920, 1080, 990, 556.875], [1920, 1080, 342, 192.375], [1000, 1000, 800, 450], [1920, 1080, 0, 0], [640, 360, 0, 0], [1920, 1080, 342, 192.375, 16]].map(function (a) {
          var z = BC.labelSize(a[0], a[1], a[2], a[3], a[4]);
          var r = function (x) { return Math.round(x * 1000) / 1000; };
          return z.scale ? [r(z.size * z.scale), r(z.halo * z.scale), r(z.gap * z.scale), r(z.scale)] : [z.size, r(z.halo), z.scale];
        })"#,
    );
    assert_eq!(
        v[0],
        json!([13, 3, 4, 0.516]),
        "1920 x 1080 at 1440 px wide"
    );
    assert_eq!(
        v[1],
        json!([13, 3, 4, 0.178]),
        "1920 x 1080 on a 390 px phone"
    );
    assert_eq!(
        v[2],
        json!([13, 3, 4, 0.45]),
        "a square frame letterboxed: the height fits"
    );
    assert_eq!(v[3], json!([34, 7.846, 0]), "unmeasured: 1080 / 32");
    assert_eq!(v[4], json!([12, 2.769, 0]), "at least 12");
    assert_eq!(v[5], json!([16, 3.692, 4.923, 0.178]), "another size");
}

/// Train -> Models, "Use as default" (assets/training.js on the stub page): a low-confidence
/// suggestion (`suggestion.confirm`) asks in the page before the form saves its threshold (the
/// box ticked, or always when the model is the default already), then posts `confirm=1` with
/// `use_threshold=1`; Cancel posts nothing; with enough data nothing is asked.
#[test]
fn the_use_form_confirms_a_low_confidence_threshold() {
    if !have_node() {
        return;
    }
    run_page(
        r##"
(async () => {
  // Train -> Models, "Use as default" with a low-confidence suggestion: the form asks in the page
  // and then posts confirm=1 with use_threshold=1; without the tick nothing is asked.
  const preview = (already, confirm) => ({already_default: already, current: 'old', new: 'new', compare: null, trained_on: null,
    classes: {lost: [], unknown: null, gained: [], gained_with_copy: [], current: 'person', new: 'person'},
    rows: [{key: 'confidence_threshold', label: 'Confidence', current: '0.50', new: '0.50', with_copy: '0.50'}],
    copy: {copied: [], not_copied: []},
    suggestion: {threshold: 0.31, device: null, words: 'At 0.31 ...', notes: [], confirm, low_confidence: confirm ? 'low confidence: 3 camera days; 4 are needed.' : undefined}});
  const run = async (already, confirm) => {
    let form, host, row;
    const W = page({path: '/train/models', strict: true,
      qsa: (sel) => sel === 'form[data-switch]' ? [form] : [],
      qs: (sel) => /tr-switch/.test(sel) ? host : null,
      route: req => /switch-preview/.test(req.url) ? {json: preview(already, confirm)} : {hold: true}});
    form = new W.El('form'); form.dataset['switch'] = 'run1';
    host = new W.El('td'); row = new W.El('tr'); row.hidden = true; row.appendChild(host); host.closest = () => row;
    W.load('training.js'); await flush();
    form.dispatch('submit'); await flush();
    const drawn = host.all().filter(e => e.tagName === 'FORM')[0];
    assert.ok(drawn, 'the preview drew its form');
    return {W, drawn, ask: drawn.all().filter(e => e.getAttribute && e.getAttribute('role') === 'alertdialog')[0],
      button: t => drawn.all().filter(e => e.tagName === 'BUTTON' && e.textContent === t)[0],
      tick: drawn.all().filter(e => e.tagName === 'INPUT' && e.name === 'use_threshold' && e.type === 'checkbox')[0]};
  };
  const has = (fields, k, v) => fields.some(f => f[0] === k && (v === undefined || f[1] === v));
  { // Not the default yet: ticked, the submit asks first; Save anyway posts confirm=1.
    const P = await run(false, true);
    assert.ok(P.ask && P.ask.hidden, 'a hidden question');
    P.drawn.requestSubmit(); assert.strictEqual(P.W.posted.length, 1, 'unticked: no question, no threshold');
    assert.ok(!has(P.W.posted[0], 'use_threshold') && !has(P.W.posted[0], 'confirm'));
    P.tick.checked = true; P.tick.dispatch('change');
    P.drawn.requestSubmit();
    assert.strictEqual(P.W.posted.length, 1, 'ticked: held for the question');
    assert.ok(!P.ask.hidden && /is low confidence: 3 camera days; 4 are needed\. Save it anyway\?/.test(P.ask.textContent) && /3 camera days/.test(P.ask.textContent), 'asks, naming why: ' + P.ask.textContent);
    P.button('Cancel').click(); assert.ok(P.ask.hidden); assert.strictEqual(P.W.posted.length, 1);
    P.drawn.requestSubmit(); P.button('Save anyway').click();
    assert.strictEqual(P.W.posted.length, 2);
    const f = P.W.posted[1];
    assert.ok(has(f, 'use_threshold', '1') && has(f, 'confirm', '1') && has(f, 'id', 'run1'), JSON.stringify(f));
  }
  { // Already the default: the form always saves the threshold, so it always asks.
    const P = await run(true, true);
    P.drawn.requestSubmit(); assert.strictEqual(P.W.posted.length, 0); assert.ok(!P.ask.hidden);
    P.button('Save anyway').click();
    assert.ok(has(P.W.posted[0], 'use_threshold', '1') && has(P.W.posted[0], 'confirm', '1'), JSON.stringify(P.W.posted));
  }
  { // Enough data: no question, no confirm.
    const P = await run(true, false);
    assert.ok(!P.ask, 'no question drawn');
    P.drawn.requestSubmit();
    assert.ok(has(P.W.posted[0], 'use_threshold', '1') && !has(P.W.posted[0], 'confirm'), JSON.stringify(P.W.posted));
  }
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##,
    );
}

/// The texts of the server's newer fields: the shell's "labels changed" banner (frames that
/// differ, the frames it could not read, the server's message when results and labels were saved
/// apart; hidden when nothing differs or it is not known yet), the Re-score result (unreadable
/// files keep their figures), a Compare row against the best, and how a pick was validated
/// (partly in-sample when a fold borrowed the pick on every frame).
#[test]
fn texts_of_labels_rescore_best_and_validation() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"({
          labels: [null, {frames: 0, changed: 0, left: 0, joined: 0}, {frames: 3, changed: 2, left: 1, joined: 0, unknown: 2},
            {frames: 50, stale: true, message: "the labels of Your cameras are not the ones these results were saved with: re-score them"},
            {frames: null, computing: true}, {frames: null, folderMissing: true, message: "gone"}, {frames: 0, unknown: 1}, {frames: 1, joined: 1}].map(BC.labelsText),
          rescored: [BC.rescoredText({frames: 50, runs: 2, needRun: 3, unknown: ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], notRescored: ["m on ort:cpu"]}), BC.rescoredText({frames: 1, runs: 1, unknown: []})],
          best: [null, {value: -0.05, low: -0.08, high: -0.02}, {value: -0.01, low: -0.04, high: 0.02}, {value: -0.03, low: null, high: null}].map(BC.vsBestText),
          cv: [{folds: 0, lowConfidence: "too little"}, {folds: 5, groups: 20, set: "cameras", lowConfidence: "low confidence: a fold borrowed", objectiveName: "F1", suggested: {metric: {value: 0.44, low: 0.35, high: 0.5}}},
            {folds: 5, groups: 20, set: "coco", objectiveName: "F1", suggested: {metric: {value: 0.44, low: 0.35, high: 0.5}}}, {folds: 5, groups: 20, set: "coco", suggested: {}}].map(BC.cvText),
          facts: [BC.cvFact(0, null), BC.cvFact(5, "x"), BC.cvFact(5, null)]
        })"#,
    );
    assert_eq!(
        v["labels"],
        json!([
            null,
            null,
            "Labels of 3 frames of Your cameras differ from this run's: 2 changed, 1 no longer reviewed. 2 frames could not be read this time; a re-score keeps their figures.",
            "The labels of Your cameras are not the ones these results were saved with: re-score them.",
            null,
            null,
            null,
            "Labels of 1 frame of Your cameras differ from this run's: 1 newly reviewed (a run scores those)."
        ])
    );
    assert_eq!(
        v["rescored"],
        json!([
            "Re-scored 50 frames of 2 runs against the current labels. Frames reviewed since the run need a new run. 4 frames could not be read (a.jpg, b.jpg, c.jpg, and 1 more) and keep their old figures; re-score again once they read. Not re-scored (no stored detections): m on ort:cpu.",
            "Re-scored 1 frame of 1 run against the current labels."
        ])
    );
    assert_eq!(
        v["best"],
        json!([
            null,
            "clearly below the best by 0.050 (95% interval of the difference −0.080 to −0.020)",
            "not clearly different from the best: −0.010 (95% interval of the difference −0.040 to +0.020)",
            "−0.030 against the best (no interval: too few groups to tell)"
        ])
    );
    assert_eq!(
        v["cv"],
        json!([
            "In-sample pick: counted on the frames it was picked on.",
            "Cross-validated over 20 camera days (5 folds), partly in-sample: F1 0.44 ±0.09.",
            "Cross-validated over 20 groups (5 folds): F1 0.44 ±0.09 on frames the pick never saw.",
            "Cross-validated over 20 groups (5 folds)."
        ])
    );
    assert_eq!(
        v["facts"],
        json!([
            "in-sample (too little data)",
            "5 folds, partly in-sample",
            "5 folds"
        ])
    );
}

/// The frame dialog (bench-frames.js on the stub page) places its labels inside the image, clear
/// of each other and each by its box, for boxes at the top edge, side by side and at the right
/// edge (they were all at the top edge, over each other, and past the right edge).
#[test]
fn the_frame_labels_stay_inside_and_apart() {
    if !have_node() {
        return;
    }
    run_page(
        r##"
(async () => {
  // The frame dialog's labels on a 2688 x 1520 frame shown 800 x 450: boxes at the top edge, side
  // by side, and at the right edge. Every label inside the image, none over another, each by its box.
  const W = page({path: '/benchmark/images', step: 'images', route: () => ({hold: true})});
  W.load('bench-core.js'); W.load('bench-frames.js');
  const boxes = [[10, 5, 300, 200, 'person'], [150, 8, 600, 250, 'vehicle'], [2500, 10, 2680, 300, 'truck'], [2600, 600, 2688, 700, 'dog'], [1000, 800, 1200, 1100, 'cat'], [1010, 790, 1190, 1000, 'bird']];
  const f = {set: 's', file: 'a.jpg', width: 2688, height: 1520, gt: boxes.map(b => ({label: b[4], bbox: b.slice(0, 4), ok: false})), preds: []};
  W.ctx.BenchFrames.open(f, {threshold: 0.3}); await flush();
  const texts = W.$('img-big').querySelectorAll('text');
  assert.strictEqual(texts.length, boxes.length, 'one label per box');
  const rects = texts.map((t, i) => {
    const size = parseFloat(t.getAttribute('font-size')), x = parseFloat(t.getAttribute('x')), y = parseFloat(t.getAttribute('y'));
    const w = t.textContent.length * size * 0.6;
    const hh = parseFloat(t.getAttribute('stroke-width')) / 2; return {i, x1: x - hh, x2: x + w + hh, y1: y - size * 0.95 - hh, y2: y + size * 0.25 + hh, size, text: t.textContent};
  });
  rects.forEach(r => {
    assert.ok(r.x1 >= 0 && r.x2 <= 2688 && r.y1 >= 0 && r.y2 <= 1520, 'inside the image: ' + JSON.stringify(r));
    const b = boxes[r.i];
    assert.ok(r.x2 > b[0] && r.x1 < b[2] && r.y2 > b[1] - 2 * r.size && r.y1 < b[3] + 2 * r.size, 'by its box: ' + JSON.stringify([r, b]));
  });
  rects.forEach(a => rects.forEach(b => {
    if (a.i < b.i) assert.ok(!(a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1), 'labels overlap: ' + JSON.stringify([a, b]));
  }));
  console.log('ok');
})().catch(e => { console.error(e); process.exit(1); });
"##,
    );
}

/// `placeBoxLabels`: above a box with room, inside its top edge at the image's top, shifted left
/// at the right edge (halo included), a line lower inside the box when the spot is taken, below the box when the
/// box is too short, and the first spot inside the image when nothing is free.
#[test]
fn box_labels_clamp_and_avoid_each_other() {
    if !have_node() {
        return;
    }
    let v = eval(
        r#"(function () {
          var o = {w: 1000, h: 500, size: 20, gap: 6};
          var r = function (p) { return [Math.round(p.x1), Math.round(p.y1), Math.round(p.x2), Math.round(p.y2)]; };
          return {
            above: r(BC.placeBoxLabels([{x1: 100, y1: 200, x2: 300, y2: 400, text: "dog", width: 60}], o)[0]),
            top: r(BC.placeBoxLabels([{x1: 100, y1: 2, x2: 300, y2: 400, text: "dog", width: 60}], o)[0]),
            right: r(BC.placeBoxLabels([{x1: 950, y1: 200, x2: 1000, y2: 300, text: "vehicle missed", width: 200}], o)[0]),
            halo: r(BC.placeBoxLabels([{x1: 950, y1: 200, x2: 1000, y2: 300, text: "v", width: 200}], {w: 1000, h: 500, size: 20, gap: 6, halo: 4})[0]),
            estimate: r(BC.placeBoxLabels([{x1: 0, y1: 200, x2: 100, y2: 300, text: "abcde"}], o)[0]),
            pair: BC.placeBoxLabels([{x1: 10, y1: 0, x2: 300, y2: 200, text: "a", width: 150}, {x1: 100, y1: 0, x2: 400, y2: 200, text: "b", width: 150}], o).map(r),
            short: BC.placeBoxLabels([{x1: 10, y1: 0, x2: 300, y2: 30, text: "a", width: 150}, {x1: 20, y1: 0, x2: 300, y2: 30, text: "b", width: 150}], o).map(r),
            full: BC.placeBoxLabels([0, 1, 2, 3, 4].map(function () { return {x1: 0, y1: 0, x2: 200, y2: 26, text: "x", width: 150}; }), {w: 200, h: 52, size: 20, gap: 6}).map(r)
          };
        })()"#,
    );
    assert_eq!(
        v["above"],
        json!([103, 170, 163, 194]),
        "above, a gap over the box"
    );
    assert_eq!(
        v["top"],
        json!([103, 5, 163, 29]),
        "no room above: inside the top edge"
    );
    assert_eq!(
        v["right"],
        json!([800, 170, 1000, 194]),
        "shifted left to end at the right edge"
    );
    assert_eq!(
        v["halo"],
        json!([796, 166, 1000, 194]),
        "the halo stays inside the image and clear of the box"
    );
    assert_eq!(
        v["estimate"],
        json!([3, 170, 63, 194]),
        "unmeasured: 0.6 x size a character"
    );
    assert_eq!(
        v["pair"],
        json!([[13, 3, 163, 27], [103, 30, 253, 54]]),
        "the second a line lower inside its box"
    );
    assert_eq!(
        v["short"],
        json!([[13, 3, 163, 27], [23, 36, 173, 60]]),
        "a short box: below it"
    );
    assert_eq!(
        v["full"][4],
        json!([3, 3, 153, 27]),
        "nothing free: the first spot inside the image"
    );
}
