//! Tests of the history file: appends and reads, torn lines, a new attempt, the move out of an
//! older `runs.json`, a file that cannot be written, and the bytes a long run writes.

use super::*;
use crate::training::progress::TrainEvent;
use crate::training::service::{
    PROGRESS_SAVE_EVERY, RUNS_FILE, RunState, TrainingService, load_records, progress_save_due,
};
use std::time::Duration;

/// A scratch folder removed on drop (kept with `BOP_KEEP_TEST_DIRS=1`).
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let d = std::env::temp_dir().join(format!("bop-history-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Self(d)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::env::var_os("BOP_KEEP_TEST_DIRS").is_none() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// An epoch event with `n` metrics whose values move with the epoch.
fn epoch_event(e: u32, epochs: u32, n: usize) -> TrainEvent {
    let mut metrics: BTreeMap<String, f64> = (0..n)
        .map(|i| {
            (
                format!("train/term_{i:02}"),
                1.0 / f64::from(e) + i as f64 * 1e-3,
            )
        })
        .collect();
    metrics.insert("lr".into(), 1e-4 * 0.99f64.powi(e as i32));
    TrainEvent::Epoch {
        epoch: e,
        epochs,
        loss: Some(3.0 / f64::from(e)),
        map50: Some(0.5 + f64::from(e) * 1e-3),
        map: Some(0.3 + f64::from(e) * 1e-3),
        elapsed_s: Some(f64::from(e) * 120.0),
        metrics,
    }
}

fn eval_event(e: u32) -> TrainEvent {
    TrainEvent::Eval {
        epoch: e,
        val_loss: Some(2.0 / f64::from(e)),
        metrics: [("f2".to_string(), 0.6 + f64::from(e) * 1e-3)]
            .into_iter()
            .collect(),
    }
}

/// `h` as a reader of any JSON copy gets it: serde_json parses floats to the nearest value
/// within an ulp, as it did the history in `runs.json`.
fn json(h: &[EpochPoint]) -> Vec<EpochPoint> {
    serde_json::from_str(&serde_json::to_string(h).unwrap()).unwrap()
}

#[test]
fn appended_rows_read_back_as_the_progress_history() {
    let t = Scratch::new("rows");
    let mut log = HistoryLog::new(&t.0);
    let mut tp = TrainProgress::default();
    assert!(!log.sync(&tp), "nothing yet");
    assert!(
        !history_path(&t.0).exists(),
        "no file before the first epoch"
    );
    for e in 1..=4 {
        tp.apply(&epoch_event(e, 10, 5), f64::from(e) * 120.0);
        assert!(log.sync(&tp));
        assert!(!log.sync(&tp), "an unchanged history writes nothing");
        tp.apply(&eval_event(e), f64::from(e) * 121.0);
        assert!(log.sync(&tp));
    }
    // A late eval of an earlier epoch changes that row only.
    tp.apply(
        &TrainEvent::Eval {
            epoch: 2,
            val_loss: None,
            metrics: [("iou".to_string(), 0.7)].into_iter().collect(),
        },
        600.0,
    );
    assert!(log.sync(&tp));
    assert_eq!(read(&history_path(&t.0)), json(&tp.history));
    assert_eq!(*cached(&history_path(&t.0)), json(&tp.history));
    // The eval lines hold what they added, not the whole row again.
    let text = std::fs::read_to_string(history_path(&t.0)).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 9, "{text}");
    assert!(!lines[1].contains("train/term_00"), "{}", lines[1]);
    assert!(lines[1].contains("val_loss"), "{}", lines[1]);
    assert_eq!(log.bytes(), text.len() as u64);
    // The cache sees the next append.
    tp.apply(&epoch_event(5, 10, 5), 700.0);
    log.sync(&tp);
    assert_eq!(cached(&history_path(&t.0)).len(), 5);
}

#[test]
fn a_torn_line_costs_only_itself() {
    let t = Scratch::new("torn");
    let path = history_path(&t.0);
    let mut log = HistoryLog::new(&t.0);
    let mut tp = TrainProgress::default();
    for e in 1..=2 {
        tp.apply(&epoch_event(e, 10, 3), 1.0);
        log.sync(&tp);
    }
    // A crash mid-append: the line is cut, with no newline.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(br#"{"epoch":3,"elapsed_s":3.0,"metrics":{"lo"#);
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(read(&path), json(&tp.history), "the torn row is left out");
    // The next append starts on a line of its own.
    let mut next = HistoryLog::new(&t.0);
    next.opened = true;
    next.written = tp.history.clone();
    tp.apply(&epoch_event(3, 10, 3), 3.0);
    next.sync(&tp);
    assert_eq!(read(&path), json(&tp.history));
    // Damaged bytes cost their line.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(b"\xff\xfe\n");
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(read(&path), json(&tp.history));
}

#[test]
fn a_new_attempt_starts_the_file_over() {
    let t = Scratch::new("attempt");
    let path = history_path(&t.0);
    // Left by an earlier run in this folder: the job's first write replaces it.
    std::fs::write(
        &path,
        "{\"epoch\":9,\"elapsed_s\":1.0,\"metrics\":{\"loss\":1.0}}\n",
    )
    .unwrap();
    let mut log = HistoryLog::new(&t.0);
    let mut tp = TrainProgress::default();
    for e in 1..=3 {
        tp.apply(&epoch_event(e, 10, 2), 1.0);
        log.sync(&tp);
    }
    assert_eq!(read(&path), json(&tp.history));
    // Out of memory: the job starts a new progress from epoch 1.
    let mut again = TrainProgress::default();
    assert!(log.sync(&again), "an emptied history is a change");
    assert!(read(&path).is_empty());
    again.apply(&epoch_event(1, 10, 2), 1.0);
    log.sync(&again);
    assert_eq!(read(&path), json(&again.history));
    // A history that went back past rows it had also restarts (no empty report between).
    let mut third = TrainProgress::default();
    for e in 1..=3 {
        third.apply(&epoch_event(e, 10, 2), 1.0);
    }
    log.sync(&third);
    let mut fourth = TrainProgress::default();
    fourth.apply(&epoch_event(1, 10, 4), 1.0);
    log.sync(&fourth);
    assert_eq!(read(&path), json(&fourth.history));
}

#[test]
fn rows_past_the_cap_stay_in_the_file_and_readers_keep_the_last() {
    let t = Scratch::new("cap");
    let mut log = HistoryLog::new(&t.0);
    let mut tp = TrainProgress::default();
    for e in 1..=(MAX_HISTORY as u32 + 3) {
        tp.apply(&epoch_event(e, MAX_HISTORY as u32 + 3, 0), 1.0);
        log.sync(&tp);
    }
    assert_eq!(tp.history.len(), MAX_HISTORY);
    assert_eq!(read(&history_path(&t.0)), json(&tp.history));
    assert_eq!(
        log.written.len(),
        MAX_HISTORY,
        "the mirror follows the progress"
    );
}

#[test]
fn a_file_that_cannot_be_written_keeps_the_history_on_the_record() {
    let t = Scratch::new("unwritable");
    // The run folder is a file: nothing can be created in it.
    let run_dir = t.0.join("run");
    std::fs::write(&run_dir, "not a folder").unwrap();
    let mut log = HistoryLog::new(&run_dir);
    let mut tp = TrainProgress::default();
    tp.apply(&epoch_event(1, 10, 2), 1.0);
    assert!(log.stored(&tp).history.is_empty(), "nothing failed yet");
    log.sync(&tp);
    assert!(log.failed());
    tp.apply(&epoch_event(2, 10, 2), 2.0);
    log.sync(&tp);
    assert_eq!(log.stored(&tp), tp, "the record keeps everything");
    assert_eq!(log.bytes(), 0);
}

#[test]
fn the_stored_progress_drops_only_the_lists() {
    let mut tp = TrainProgress::default();
    for e in 1..=3 {
        tp.apply(&epoch_event(e, 10, 2), 1.0);
    }
    let bare = tp.without_history();
    assert!(bare.history.is_empty() && bare.losses.is_empty());
    let mut back = bare.clone();
    back.history = tp.history.clone();
    back.losses = tp.losses.clone();
    assert_eq!(back, tp);
}

/// The fixture: a record of `runs.json` as it was written before the history file (5 epochs),
/// its run folder at `run_dir`.
fn old_runs_json(run_dir: &Path) -> String {
    let dir = serde_json::to_string(&run_dir.to_string_lossy()).unwrap();
    include_str!("../../../../tests/data/runs-v0-with-history.json")
        .replace("\"@RUN_DIR@", &dir[..dir.len() - 1])
}

fn fixture_history() -> Vec<EpochPoint> {
    let v: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/data/runs-v0-with-history.json"
    ))
    .unwrap();
    serde_json::from_value(v[0]["progress"]["history"].clone()).unwrap()
}

fn service(store: &Path) -> Arc<TrainingService> {
    let fake = Arc::new(crate::resources::export::fake::Fake::new());
    TrainingService::start(
        fake.clone(),
        fake,
        Arc::new(crate::training::job::NoHooks),
        Some(store.to_path_buf()),
    )
}

#[test]
fn an_old_runs_json_loads_shows_and_moves_its_history_to_the_file() {
    let t = Scratch::new("migrate");
    let run_dir = t.0.join("20261028-011025-3bf3");
    std::fs::create_dir_all(&run_dir).unwrap();
    let store = t.0.join(RUNS_FILE);
    std::fs::write(&store, old_runs_json(&run_dir)).unwrap();
    let want = fixture_history();
    assert_eq!(want.len(), 5);
    // A reader that does not save shows it as it is, and writes nothing.
    let before = std::fs::read(&store).unwrap();
    let shown = load_records(&store);
    assert_eq!(shown[0].progress.as_ref().unwrap().history, want);
    assert_eq!(std::fs::read(&store).unwrap(), before);
    assert!(!history_path(&run_dir).exists());

    let svc = service(&store);
    let r = svc.get("20261028-011025-3bf3").unwrap();
    assert_eq!(r.state, RunState::Done);
    let p = r.progress.as_ref().unwrap();
    assert_eq!(p.history, want, "shown with every epoch");
    assert_eq!(p.losses.len(), 5);
    assert_eq!(p.best.as_ref().unwrap().epoch, 1);
    assert_eq!(svc.runs()[0], r);
    // The file holds the epochs; runs.json holds them no more.
    assert_eq!(read(&history_path(&run_dir)), want);
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&store).unwrap()).unwrap();
    assert_eq!(saved[0]["progress"]["history"], serde_json::json!([]));
    assert_eq!(saved[0]["progress"]["best"]["epoch"], 1);
    let saved_len = std::fs::read(&store).unwrap().len();
    assert!(
        saved_len * 2 < before.len(),
        "{saved_len} of {}",
        before.len()
    );
    drop(svc);
    // Every reader still shows them, and a second start changes nothing.
    assert_eq!(load_records(&store)[0], r);
    let file = std::fs::read(history_path(&run_dir)).unwrap();
    let svc = service(&store);
    assert_eq!(svc.get(&r.id).unwrap(), r);
    assert_eq!(std::fs::read(history_path(&run_dir)).unwrap(), file);
    assert_eq!(std::fs::read(&store).unwrap().len(), saved_len);
}

#[test]
fn a_move_cut_short_merges_with_the_file_already_there() {
    let t = Scratch::new("merge");
    let run_dir = t.0.join("run");
    std::fs::create_dir_all(&run_dir).unwrap();
    let store = t.0.join(RUNS_FILE);
    std::fs::write(&store, old_runs_json(&run_dir)).unwrap();
    // A file from a move that wrote it but crashed before saving runs.json, plus a line the
    // record never had.
    let want = fixture_history();
    let mut extra = want[1].clone();
    extra.metrics = [("late".to_string(), 1.5)].into_iter().collect();
    std::fs::write(
        history_path(&run_dir),
        lines_of(&[want[0].clone(), want[1].clone(), extra]),
    )
    .unwrap();
    let svc = service(&store);
    let shown = svc.runs()[0].progress.clone().unwrap().history;
    assert_eq!(shown.len(), 5);
    assert_eq!(shown[1].metrics["late"], 1.5);
    let mut without = shown.clone();
    without[1].metrics.remove("late");
    assert_eq!(without, want);
}

#[test]
fn a_run_whose_folder_is_gone_keeps_its_history_in_runs_json() {
    let t = Scratch::new("gone");
    let store = t.0.join(RUNS_FILE);
    std::fs::write(&store, old_runs_json(&t.0.join("deleted-by-hand"))).unwrap();
    let svc = service(&store);
    assert_eq!(
        svc.runs()[0].progress.clone().unwrap().history,
        fixture_history()
    );
    assert!(
        !t.0.join("deleted-by-hand").exists(),
        "no folder made for it"
    );
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&store).unwrap()).unwrap();
    assert_eq!(
        saved[0]["progress"]["history"].as_array().unwrap().len(),
        5,
        "not saved, so still in the file"
    );
}

/// Bytes a 300-epoch run of 10 hours writes, before (the record rewritten every 30 s with its
/// whole history) and now (the record every 5 minutes, the history appended per change). One run
/// in `runs.json`: every other record was rewritten with it before, so the old figure is a floor.
#[test]
fn bytes_written_by_a_300_epoch_run() {
    let t = Scratch::new("bytes");
    let run_dir = t.0.join("run");
    let template: Vec<RunRecord> = serde_json::from_str(&old_runs_json(&run_dir)).unwrap();
    let record = |p: TrainProgress| {
        let mut r = template[0].clone();
        r.state = RunState::Running;
        r.progress = Some(p);
        serde_json::to_vec_pretty(&[r]).unwrap().len() as u64
    };
    let (epochs, epoch_s) = (300u32, 120u64);
    let mut results = Vec::new();
    for metrics in [36usize, 64] {
        let mut tp = TrainProgress::default();
        let mut log = HistoryLog::new(&run_dir);
        let (mut old, mut old_saves, mut new, mut new_saves) = (0u64, 0u32, 0u64, 0u32);
        let (mut old_last, mut new_last) = (0u64, 0u64);
        let mut sizes: HashMap<u64, u64> = HashMap::new();
        // One progress report a second (a batch event); the epoch and its eval at its end.
        for s in 0..u64::from(epochs) * epoch_s {
            if s > 0 && s % epoch_s == 0 {
                let e = u32::try_from(s / epoch_s).unwrap();
                tp.apply(&epoch_event(e, epochs, metrics - 4), s as f64);
                tp.apply(&eval_event(e), s as f64);
            }
            let new_stage = s == 0;
            if new_stage || s - old_last >= 30 {
                // The progress changes at epoch ends only: one serialization per state.
                old += *sizes
                    .entry(tp.history_rev.0)
                    .or_insert_with(|| record(tp.clone()));
                old_saves += 1;
                old_last = s;
            }
            log.sync(&tp);
            if progress_save_due(new_stage, Duration::from_secs(s - new_last)) {
                new += record(log.stored(&tp));
                new_saves += 1;
                new_last = s;
            }
        }
        assert!(!log.failed());
        let file = std::fs::metadata(history_path(&run_dir)).unwrap().len();
        assert_eq!(log.bytes(), file);
        assert_eq!(read(&history_path(&run_dir)), json(&tp.history));
        let per_epoch = tp.history.first().map_or(0, |p| p.metrics.len());
        results.push((per_epoch, old, old_saves, new, new_saves, file));
        // At least 100 times fewer bytes.
        assert!((new + file) * 100 < old, "{results:?}");
    }
    assert_eq!(PROGRESS_SAVE_EVERY, Duration::from_secs(300));
    for (m, old, os, new, ns, file) in results {
        eprintln!(
            "300 epochs x {m} metrics, 10 h: before {old} B in {os} saves of runs.json; now \
             {new} B in {ns} saves + {file} B of history.jsonl = {} B",
            new + file
        );
    }
}

/// A run stopped by a crash: its record was saved before the last epochs (the progress is saved
/// every 5 minutes), its history file at every epoch. The best epoch comes from the file.
#[test]
fn an_interrupted_run_takes_its_best_epoch_from_the_file() {
    let t = Scratch::new("interrupted");
    let run_dir = t.0.join("run");
    std::fs::create_dir_all(&run_dir).unwrap();
    let store = t.0.join(RUNS_FILE);
    let mut runs: Vec<RunRecord> = serde_json::from_str(&old_runs_json(&run_dir)).unwrap();
    let mut epochs = fixture_history();
    epochs[4].metrics.insert("map".into(), 0.99);
    epochs.push(EpochPoint {
        epoch: 6,
        elapsed_s: 600.0,
        metrics: [("map".to_string(), 0.5)].into_iter().collect(),
    });
    std::fs::write(history_path(&run_dir), lines_of(&epochs)).unwrap();
    let p = runs[0].progress.as_mut().unwrap();
    // Saved at epoch 3, before the best one.
    *p = p.without_history();
    p.stopped = None;
    p.phase = None;
    p.best.as_mut().unwrap().epoch = 1;
    runs[0].state = RunState::Running;
    runs[0].output = None;
    runs[0].registered = None;
    std::fs::write(&store, serde_json::to_vec_pretty(&runs).unwrap()).unwrap();
    let svc = service(&store);
    let r = svc.get(&runs[0].id).unwrap();
    assert_eq!(r.state, RunState::Failed);
    let p = r.progress.unwrap();
    assert_eq!(p.best.as_ref().map(|b| (b.epoch, b.value)), Some((5, 0.99)));
    assert_eq!(p.since_best, 1);
    assert_eq!(p.history.len(), 6);
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&store).unwrap()).unwrap();
    assert_eq!(saved[0]["progress"]["history"], serde_json::json!([]));
}
