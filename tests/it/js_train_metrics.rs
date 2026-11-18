//! The Train tab's run metrics (`assets/train-metrics.js`, `TrainMetrics`) and the support rule:
//! `ruleKey` is the twin of `TrainProgress::rule_key` (what the best epoch and the convergence rule
//! follow), `selectedBy` names it ("by F2 over car, person"), `classRows` reads the per-class values
//! of the best epoch, and `render` draws the per-class table (cards on a phone, "too few to judge"
//! below the support event's `min_boxes`) and leads the per-epoch table with the selected average.
//! Run in `node` with a stub DOM.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).
#![allow(clippy::unwrap_used)]

use blue_onyx_prism::training::progress::{self, TrainProgress};
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

/// A stub DOM big enough for train-metrics.js without charts (no `PrismBench`): `run(cases)`
/// is the driver's body; it prints `RESULT: <json>`.
const STUB: &str = r#"
const fs = require('fs'), vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const input = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
class N {
  constructor(tag) { this.tagName = tag; this.className = ''; this.attrs = {}; this.children = []; this.parentNode = null; this._text = ''; this.listeners = {}; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this.children.length ? this.children.map(c => c.textContent).join('') : this._text; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  replaceChildren(...cs) { this.children = []; cs.forEach(c => this.appendChild(c)); }
  replaceChild(n, o) { const i = this.children.indexOf(o); this.children[i] = n; n.parentNode = this; return o; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  get childNodes() { return this.children; }
  all(pred, out = []) { if (pred(this)) out.push(this); this.children.forEach(c => c.all(pred, out)); return out; }
}
const document = { createElement: t => new N(t), createTextNode: t => { const n = new N('#text'); n.textContent = t; return n; } };
const window = { document };
const ctx = { window, document, globalThis: {} };
vm.createContext(ctx);
vm.runInContext(src, ctx);
const T = ctx.window.TrainMetrics;
const hasClass = (n, c) => (n.className || '').split(' ').indexOf(c) >= 0;
"#;

/// Run `body` after [`STUB`] with `input` (JSON); the `RESULT: ` line's JSON.
fn node(body: &str, input: &Value) -> Value {
    let dir = std::env::temp_dir().join(format!("bop-js-train-metrics-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let driver = dir.join("driver.js");
    let data = dir.join("input.json");
    std::fs::write(&driver, format!("{STUB}\n{body}")).unwrap();
    std::fs::write(&data, input.to_string()).unwrap();
    let asset = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/train-metrics.js");
    let out = Command::new("node")
        .arg(&driver)
        .arg(&asset)
        .arg(&data)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "node failed ({}): {stdout}{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let line = stdout
        .lines()
        .find_map(|l| l.strip_prefix("RESULT: "))
        .unwrap_or_else(|| panic!("no result in {stdout}"));
    serde_json::from_str(line).unwrap()
}

fn support(key: &str, fell_back: bool) -> Value {
    json!({"min_boxes": 20, "boxes": {"car": 428, "person": 34, "truck": 7, "bicycle": 1},
        "classes": ["car", "person"], "fell_back": fell_back, "key": key, "reason": null})
}

fn settings(select: &str) -> Value {
    json!({"lr": null, "lr_encoder": null, "lr_schedule": "cosine", "lr_min_factor": 0.1,
        "select_metric": select, "epochs": 50, "patience": 10, "patience_doubled": false,
        "batch": 4, "workers": null, "seed": null, "scene": null, "gpu": null})
}

/// Progress records over the support rule's cases: selected metric, support event (supported,
/// fell back, a key for another metric, none), a fall-back.
fn cases() -> Vec<Value> {
    let mut out = Vec::new();
    for select in ["map", "map50", "f1", "f2", "f0_5", "iou"] {
        let sup_key = format!("{select}_sup");
        for sup in [
            Some(support(&sup_key, false)),
            Some(support(&sup_key, true)),
            Some(support("f2_sup", false)),
            Some(support(select, false)),
            None,
        ] {
            for fallback in [false, true] {
                let mut p = json!({"epoch": 3, "epochs": 50, "settings": settings(select)});
                if let Some(s) = &sup {
                    p["support"] = s.clone();
                }
                if fallback {
                    p["select_fallback"] = json!({"metric": sup_key, "epoch": 2, "reason": "the per-class metrics failed"});
                }
                out.push(p);
            }
        }
    }
    // No settings (a record from before them): mAP.
    out.push(json!({"epoch": 1, "support": support("map_sup", false)}));
    out.push(json!({"epoch": 1}));
    out
}

/// `TrainMetrics.ruleKey` decides as `TrainProgress::rule_key` on every case, read back from the
/// service's own serialization.
#[test]
fn rule_key_is_the_service_twin() {
    if !have_node() {
        return;
    }
    let records: Vec<Value> = cases()
        .into_iter()
        .map(|c| {
            let p: TrainProgress = serde_json::from_value(c).unwrap();
            serde_json::to_value(&p).unwrap()
        })
        .collect();
    let want: Vec<String> = records
        .iter()
        .map(|r| {
            serde_json::from_value::<TrainProgress>(r.clone())
                .unwrap()
                .rule_key()
                .to_string()
        })
        .collect();
    let got = node(
        "console.log('RESULT: ' + JSON.stringify(input.map(p => T.ruleKey(p))));",
        &json!(records),
    );
    assert_eq!(got, json!(want));
    // The cases cover every outcome.
    for k in ["map", "map_sup", "f2", "f2_sup", "f0_5_sup", "iou"] {
        assert!(want.iter().any(|w| w == k), "{k}");
    }
}

/// `selectedBy` names the averaged classes (at most four), `classRows` takes the best epoch's
/// per-class values with the support event's boxes, most first, marking the classes under
/// `min_boxes`; `render` leads the per-epoch table with the selected average and draws the
/// per-class table as cards.
#[test]
fn per_class_values_and_the_selected_average() {
    if !have_node() {
        return;
    }
    let mut many = support("f2_sup", false);
    many["classes"] = json!(["car", "person", "truck", "bicycle", "dog", "cat"]);
    let row = |e: u32, f2s: f64, car: f64| {
        json!({"epoch": e, "elapsed_s": e, "metrics": {"map": 0.4, "map50": 0.6, "f2": 0.5,
            "f2_sup": f2s, "loss": 2.0, "class/car/ap50": car, "class/car/ap": car / 2.0,
            "class/car/recall": 0.9, "class/person/ap50": 0.5, "class/truck/ap50": 0.25,
            "class/raccoon/ap50": 0.1}})
    };
    let progress = json!({"epoch": 3, "epochs": 50, "patience": 10, "since_best": 1,
        "settings": settings("f2"), "support": support("f2_sup", false),
        "best": {"metric": "f2_sup", "epoch": 2, "value": 0.8},
        "history": [row(1, 0.5, 0.7), row(2, 0.8, 0.9), row(3, 0.6, 0.8)]});
    let metrics = serde_json::to_value(progress::METRICS).unwrap();
    let input = json!({"p": progress, "many": many, "metrics": metrics,
        "fallback": {"settings": settings("f2"), "support": support("f2_sup", false),
            "select_fallback": {"metric": "f2_sup", "epoch": 2, "reason": "x"}},
        "fell": {"settings": settings("f2"), "support": support("f2", true)},
        "map": {"settings": settings("map"), "support": support("map_sup", false)},
        "plain": {"settings": settings("map")}});
    let body = r#"
const info = T.infoMap(input.metrics);
const by = ["p", "fallback", "fell", "map", "plain"].map(k => T.selectedBy(input[k], info));
by.push(T.selectedBy({settings: input.p.settings, support: input.many}, info));
const rows = T.classRows(input.p);
// No best epoch with per-class values: the newest that has them.
const late = T.classRows(Object.assign({}, input.p, {best: {epoch: 9}}));
const none = T.classRows({history: [{epoch: 1, metrics: {map: 0.5}}]});
const host = document.createElement('div');
host.dataset = {};
T.render(host, input.p, ["map", "map50", "f2", "loss"], input.metrics, {runId: "r1"});
const tables = host.all(n => n.tagName === 'table');
const head = t => t.all(n => n.tagName === 'th').map(th => th.textContent);
const cards = tables.filter(t => hasClass(t, 'rcards'))[0];
const cells = cards ? cards.all(n => n.tagName === 'tr' && n.parentNode.tagName === 'tbody').map(tr =>
  tr.children.map(td => [td.getAttribute('data-label'), td.textContent])) : null;
const tiles = host.all(n => hasClass(n, 'tr-kpi')).map(n => n.textContent);
const text = host.textContent;
console.log('RESULT: ' + JSON.stringify({by, rows, late: late && late.epoch, none, heads: tables.map(head), cells, tiles,
  note: text.indexOf('Averages count the classes with at least 20 validation boxes') >= 0}));
"#;
    let v = node(body, &input);
    assert_eq!(
        v["by"],
        json!([
            "by F2 over car, person",
            "",
            "by F2",
            "by mAP over car, person",
            "",
            "by F2 over car, person, truck, bicycle +2"
        ])
    );
    // The best epoch (2), most validation boxes first; a class the event does not list last.
    assert_eq!(v["rows"]["epoch"], 2);
    assert_eq!(v["rows"]["min"], 20);
    let rows = v["rows"]["rows"].as_array().unwrap();
    let names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["car", "person", "truck", "raccoon"]);
    assert_eq!(rows[0]["ap50"], 0.9);
    assert_eq!(rows[0]["ap"], 0.45);
    assert_eq!(rows[0]["recall"], 0.9);
    assert_eq!(rows[0]["boxes"], 428);
    let few: Vec<bool> = rows.iter().map(|r| r["few"].as_bool().unwrap()).collect();
    assert_eq!(few, [false, false, true, false]);
    assert!(rows[3]["boxes"].is_null());
    assert_eq!(v["late"], 3);
    assert!(v["none"].is_null());
    // The per-epoch table leads with the selected average (not a chosen column), then the
    // chosen ones; the per-class table is cards with labelled cells.
    let heads = v["heads"].as_array().unwrap();
    assert_eq!(heads.len(), 2, "{heads:?}");
    let short = progress::metric("f2_sup").unwrap().short;
    assert_eq!(heads[0][0], "Epoch");
    assert_eq!(heads[0][1], short);
    assert_eq!(
        heads[1],
        json!(["Class", "Val boxes", "AP50", "AP50-95", "Recall", ""])
    );
    let cells = v["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 4);
    assert_eq!(cells[0][0], json!([null, "car"]));
    assert_eq!(cells[0][1], json!(["Val boxes", "428"]));
    assert_eq!(cells[0][2], json!(["AP50", "0.900"]));
    assert_eq!(cells[2][5], json!([null, "too few to judge"]));
    assert_eq!(cells[3][1], json!(["Val boxes", "-"]));
    assert_eq!(cells[3][5], json!([null, ""]));
    assert_eq!(v["note"], true);
    // The tiles: the best names the average, the countdown what it follows.
    let tiles: Vec<&str> = v["tiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    assert!(
        tiles
            .iter()
            .any(|t| t.contains(&format!("Best {short}")) && t.contains("0.800")),
        "{tiles:?}"
    );
    assert!(
        tiles
            .iter()
            .any(|t| t.starts_with("Convergence") && t.contains("by F2 over car, person")),
        "{tiles:?}"
    );
}
