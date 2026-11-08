//! The files a 0.2.0 install leaves on disk, read by this build: the config, `runs.json`,
//! `tasks.json`, label files, sidecars, `index.jsonl` and `review-queue.jsonl` lines, and
//! `benchmark.json` (fixtures in `tests/data/0.2.0/`). Each loads with nothing refused, keeps
//! every value it had (each key of the fixture has the same value in what this build makes of it,
//! or the key is absent and its value was empty), and stays readable through a save.
//!
//! How the fixtures were made: by hand from the structs of the `0.2.0` tag (`git show
//! 0.2.0:src/config.rs`, `src/training/{service,task,job,recipe,progress,bench}.rs`,
//! `src/collect/{record,layout,config}.rs`, `src/collect/labels/mod.rs`,
//! `src/benchmark/{mod,metrics,grade,images}.rs`, `src/benchmark/report/mod.rs`), with their field
//! order, `skip_serializing_if` and enum spellings, values from the examples of 0.2.0's
//! `docs/DATASET.md`; not by a 0.2.0 binary. `@TRAINING_DIR@` in `runs.json` stands for the
//! training folder (made here). They are frozen like `tests/data/overlaps_parity.json`: they
//! show what 0.2.0 wrote, so a failure here is a change in this code, never a reason to edit them.

use crate::common::temp_dir;
use blue_onyx_prism::benchmark::BenchmarkResults;
use blue_onyx_prism::collect::labels::{self, Frame};
use blue_onyx_prism::collect::layout::{self, ReviewLine};
use blue_onyx_prism::collect::record::{IndexLine, RECORD_VERSION, Record};
use blue_onyx_prism::config::{CONFIG_FORMAT, Config};
use blue_onyx_prism::training::service::{self, RunRecord};
use blue_onyx_prism::training::task::{self, RECORD_FORMAT, TaskRecord};
use serde_json::Value;
use std::path::Path;

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/0.2.0")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn json(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

/// Every key of `old` (a fixture) is in `new` with the same value, or absent with an empty one
/// (`null`, `false`, `[]`, `{}`: what a `skip_serializing_if` drops). Numbers compare as f32
/// (most stored scores are).
fn assert_kept(old: &Value, new: &Value, at: &str) {
    match (old, new) {
        (Value::Object(o), Value::Object(n)) => {
            for (k, v) in o {
                let path = format!("{at}.{k}");
                match n.get(k) {
                    Some(nv) => assert_kept(v, nv, &path),
                    None => assert!(is_empty(v), "{path} = {v} is lost"),
                }
            }
        }
        (Value::Array(o), Value::Array(n)) => {
            assert_eq!(o.len(), n.len(), "{at}: {old} became {new}");
            for (i, (a, b)) in o.iter().zip(n).enumerate() {
                assert_kept(a, b, &format!("{at}[{i}]"));
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            assert!(
                (a - b).abs() <= 1e-6 * a.abs().max(1.0),
                "{at}: {a} became {b}"
            );
        }
        _ => assert_eq!(old, new, "{at}"),
    }
}

fn is_empty(v: &Value) -> bool {
    match v {
        Value::Null | Value::Bool(false) => true,
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

/// No `<name>.corrupt-*` file next to `path`: nothing was set aside.
fn nothing_aside(path: &Path) {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let aside: Vec<_> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with(&format!("{name}.corrupt-")))
        .collect();
    assert!(aside.is_empty(), "set aside: {aside:?}");
}

#[test]
fn config_of_0_2_0_loads_with_every_setting_and_saves_in_the_current_format() {
    let dir = temp_dir("fmt020_config");
    let path = dir.join("blue_onyx_prism_config.json");
    let text = fixture("config.json");
    std::fs::write(&path, &text).unwrap();
    let old = json(&text);
    assert!(old.get("format").is_none(), "0.2.0 wrote no format");

    let c = Config::load(&path).unwrap();
    assert_eq!(c.format, CONFIG_FORMAT);
    assert_eq!(c.models.len(), 3);
    assert!(c.collect.enabled && c.collect.hidden_classes == ["squirrel"]);
    assert_kept(&old, &serde_json::to_value(&c).unwrap(), "config");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        text,
        "a load writes nothing"
    );

    c.save(&path).unwrap();
    let saved = json(&std::fs::read_to_string(&path).unwrap());
    assert_eq!(saved["format"], CONFIG_FORMAT);
    assert_kept(&old, &saved, "saved config");
    assert_eq!(Config::load(&path).unwrap(), c);
}

/// `runs.json` with `@TRAINING_DIR@` replaced by `dir`, and every run folder made.
fn runs_json_in(dir: &Path) -> String {
    let quoted = serde_json::to_string(&dir.to_string_lossy()).unwrap();
    let text = fixture("runs.json").replace("@TRAINING_DIR@", &quoted[1..quoted.len() - 1]);
    for r in json(&text).as_array().unwrap() {
        std::fs::create_dir_all(r["run_dir"].as_str().unwrap()).unwrap();
    }
    text
}

#[test]
fn runs_json_of_0_2_0_loads_every_run_and_survives_the_history_move_and_a_save() {
    let dir = temp_dir("fmt020_runs");
    let path = dir.join(service::RUNS_FILE);
    let text = runs_json_in(&dir);
    std::fs::write(&path, &text).unwrap();
    let old = json(&text);

    let loaded = task::load_or_move_aside::<RunRecord>(&path);
    assert!(loaded.writable);
    nothing_aside(&path);
    let mut runs = loaded.records;
    assert_eq!(runs.len(), 3);
    assert_kept(&old, &serde_json::to_value(&runs).unwrap(), "runs");
    // What the page shows (no busy run here, so no state changes).
    assert_kept(
        &old,
        &serde_json::to_value(service::load_records(&path)).unwrap(),
        "shown runs",
    );

    // What the queue does at start: the epochs move to the run's history file, then a save.
    assert_eq!(service::history::migrate(&mut runs), 1);
    task::save_file(&path, &runs);
    nothing_aside(&path);
    let saved = json(&std::fs::read_to_string(&path).unwrap());
    let list = saved.as_array().expect("still a bare list");
    assert!(list.iter().all(|r| r["format"] == RECORD_FORMAT));
    assert_kept(
        &old,
        &serde_json::to_value(service::load_records(&path)).unwrap(),
        "runs after the save",
    );
    // A reader of the whole file as one list, as 0.2.0 was, still reads it.
    let whole: Vec<RunRecord> = serde_json::from_value(saved).unwrap();
    assert_eq!(whole.len(), 3);
}

#[test]
fn tasks_json_of_0_2_0_loads_every_task_and_survives_a_save() {
    let dir = temp_dir("fmt020_tasks");
    let path = dir.join(task::TASKS_FILE);
    let text = fixture("tasks.json");
    std::fs::write(&path, &text).unwrap();
    let old = json(&text);

    let loaded = task::load_or_move_aside::<TaskRecord>(&path);
    assert!(loaded.writable);
    nothing_aside(&path);
    assert_eq!(loaded.records.len(), 3);
    assert!(loaded.records.iter().all(|r| r.uses.is_empty()));
    assert_kept(
        &old,
        &serde_json::to_value(&loaded.records).unwrap(),
        "tasks",
    );

    task::save_records(&path, &loaded.records);
    let saved = json(&std::fs::read_to_string(&path).unwrap());
    assert!(
        saved
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["format"] == RECORD_FORMAT)
    );
    let back = task::load_records(&path);
    assert_eq!(back, loaded.records);
    assert_kept(&old, &saved, "saved tasks");
    nothing_aside(&path);
}

/// The two frames of the fixtures in a dataset folder: image (placeholder bytes), sidecar and
/// label file in their day folders, the day's `index.jsonl` and the review queue. Returns the
/// queue's frame paths.
fn dataset_in(dir: &Path) -> Vec<String> {
    let queue = fixture("review-queue.jsonl");
    std::fs::write(dir.join("review-queue.jsonl"), &queue).unwrap();
    let frames = [
        ("sidecar.json", "reviewed.labels.json"),
        ("sidecar-import.json", "unreviewed.labels.json"),
    ];
    let mut rels = Vec::new();
    for (line, (sidecar, labels)) in queue.lines().zip(frames) {
        let q = json(line);
        let rel = q["frame"].as_str().unwrap().to_string();
        let jpg = dir.join(&rel);
        std::fs::create_dir_all(jpg.parent().unwrap()).unwrap();
        std::fs::write(&jpg, b"\xff\xd8 not decoded here \xff\xd9").unwrap();
        std::fs::write(dir.join(q["sidecar"].as_str().unwrap()), fixture(sidecar)).unwrap();
        std::fs::write(labels::path_of_frame(&jpg), fixture(labels)).unwrap();
        rels.push(rel);
    }
    std::fs::write(
        dir.join("ir/driveway/2026-10-22").join(layout::INDEX_FILE),
        fixture("index.jsonl"),
    )
    .unwrap();
    rels
}

#[test]
fn dataset_files_of_0_2_0_load_as_frames_with_their_labels() {
    let dir = temp_dir("fmt020_dataset");
    let rels = dataset_in(&dir);

    // The review queue and the index: every line parses.
    let queue = layout::read_lines::<ReviewLine>(&dir.join("review-queue.jsonl"));
    assert_eq!(
        (queue.lines.len(), queue.damaged, queue.failed),
        (2, 0, None)
    );
    let old_queue: Vec<Value> = fixture("review-queue.jsonl").lines().map(json).collect();
    assert_kept(
        &Value::Array(old_queue),
        &serde_json::to_value(&queue.lines).unwrap(),
        "review-queue",
    );
    let index_path = dir.join("ir/driveway/2026-10-22").join(layout::INDEX_FILE);
    let index = layout::read_lines::<IndexLine>(&index_path);
    assert_eq!(
        (index.lines.len(), index.damaged, index.failed),
        (3, 0, None)
    );
    let old_index: Vec<Value> = fixture("index.jsonl").lines().map(json).collect();
    assert_kept(
        &Value::Array(old_index),
        &serde_json::to_value(&index.lines).unwrap(),
        "index",
    );
    let summary = layout::Summary::scan(&dir);
    assert_eq!((summary.frames, summary.damaged_lines), (3, 0));
    assert_eq!(summary.by_camera["driveway"].frames, 2);

    // Each queued frame resolves, with its sidecar and label file.
    let sidecars = ["sidecar.json", "sidecar-import.json"];
    let label_files = ["reviewed.labels.json", "unreviewed.labels.json"];
    for ((rel, sidecar), label_file) in rels.iter().zip(sidecars).zip(label_files) {
        let frame = Frame::resolve(&dir, rel).unwrap_or_else(|| panic!("{rel} does not resolve"));
        let rec = frame.record().unwrap();
        assert!(rec.version <= RECORD_VERSION);
        assert_kept(
            &json(&fixture(sidecar)),
            &serde_json::to_value(&rec).unwrap(),
            sidecar,
        );
        let back: Record = serde_json::from_str(&serde_json::to_string(&rec).unwrap()).unwrap();
        assert_eq!(back, rec);

        let lf = frame.labels().unwrap().expect("a label file");
        let old = json(&fixture(label_file));
        assert_kept(&old, &serde_json::to_value(&lf).unwrap(), label_file);
        labels::write(&frame.labels, &lf).unwrap();
        let again = frame.labels().unwrap().unwrap();
        assert_eq!(again, lf);
        assert_kept(
            &old,
            &json(&std::fs::read_to_string(&frame.labels).unwrap()),
            label_file,
        );
    }

    // Index and queue lines as this build writes them read back the same.
    for l in &index.lines {
        let back: IndexLine = serde_json::from_str(&serde_json::to_string(l).unwrap()).unwrap();
        assert_eq!(&back, l);
    }
    for l in &queue.lines {
        let back: ReviewLine = serde_json::from_str(&serde_json::to_string(l).unwrap()).unwrap();
        assert_eq!(&back, l);
    }
}

#[test]
fn an_empty_confirmed_label_file_of_0_2_0_reads_and_writes_back() {
    let dir = temp_dir("fmt020_empty");
    let path = dir.join("20261022T130102003Z_cd34ef56.labels.json");
    let text = fixture("empty.labels.json");
    std::fs::write(&path, &text).unwrap();
    let lf = labels::read(&path).unwrap().expect("a label file");
    assert!(lf.empty_confirmed && lf.boxes.is_empty() && lf.bulk_accepted);
    let old = json(&text);
    assert_kept(&old, &serde_json::to_value(&lf).unwrap(), "empty");
    labels::write(&path, &lf).unwrap();
    assert_eq!(labels::read(&path).unwrap().unwrap(), lf);
    assert_kept(
        &old,
        &json(&std::fs::read_to_string(&path).unwrap()),
        "empty saved",
    );
}

#[test]
fn benchmark_json_of_0_2_0_loads_every_model_and_survives_a_save() {
    let dir = temp_dir("fmt020_bench");
    let path = dir.join("benchmark.json");
    let text = fixture("benchmark.json");
    std::fs::write(&path, &text).unwrap();
    let old = json(&text);

    let (r, warnings) = BenchmarkResults::load_with_warnings(&path).unwrap();
    let r = r.expect("results");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(r.models.len(), 1);
    assert_eq!(r.models[0].devices.len(), 2);
    assert_kept(&old, &serde_json::to_value(&r).unwrap(), "benchmark");

    r.save(&path).unwrap();
    let (back, warnings) = BenchmarkResults::load_with_warnings(&path).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(back.unwrap(), r);
    assert_kept(
        &old,
        &json(&std::fs::read_to_string(&path).unwrap()),
        "saved benchmark",
    );
}

#[test]
#[should_panic(expected = "a.b = 1 is lost")]
fn assert_kept_names_a_lost_key() {
    assert_kept(&json(r#"{"b": 1, "c": null}"#), &json(r#"{"d": 1}"#), "a");
}

#[test]
#[should_panic(expected = "a.b[1]: 0.5 became 0.6")]
fn assert_kept_names_a_changed_value() {
    assert_kept(
        &json(r#"{"b": [1, 0.5]}"#),
        &json(r#"{"b": [1, 0.6], "c": 2}"#),
        "a",
    );
}

#[test]
fn assert_kept_takes_an_added_key_a_dropped_empty_one_and_f32_rounding() {
    assert_kept(
        &json(r#"{"b": 870.2, "e": [], "f": false, "g": {}, "h": null}"#),
        &json(r#"{"b": 870.2000122070312, "format": 1}"#),
        "a",
    );
}
