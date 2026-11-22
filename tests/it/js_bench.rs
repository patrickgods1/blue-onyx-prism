//! The Benchmark pages' pure rules, `assets/bench-core.js` (`BenchCore`), run in `node` without a
//! DOM: number formats, the goal catalog and its query (`?goal=f2&budget=500`, what the views and
//! `POST /v1/benchmark/goal` read), the row chips, the Compare columns' stored choice, the run's
//! per-combo states from the run poll, the CPU reference rule of the Run step, the run-time
//! estimate, old `/benchmark#...` addresses, the Images page's "disagrees with", the box
//! geometry, the CSV, and the first-load text (twin of `report::first_load_text`).
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
          text: BC.columnsText(["roc_auc", "f05", "x"]),
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
    assert_eq!(v["text"], r#"["f05","roc_auc"]"#);
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
fn disagreements_boxes_and_csv() {
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
            box: BC.boxRect([10, 20, 110, 70], 200, 100),
            flipped: BC.boxRect([110, 70, 10, 20], 200, 100),
            clamped: BC.boxRect([-10, -5, 250, 120], 200, 100),
            noSize: BC.boxRect([0, 0, 1, 1], 0, 100),
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
    assert_eq!(
        v["box"],
        json!({"left": 5, "top": 20, "width": 50, "height": 50})
    );
    assert_eq!(v["flipped"], v["box"]);
    assert_eq!(
        v["clamped"],
        json!({"left": 0, "top": 0, "width": 100, "height": 100})
    );
    assert_eq!(v["noSize"], Value::Null);
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
