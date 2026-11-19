//! [`super::learned`] rule by rule, with the dev box's measurements as the fixture (GTX 1070 Ti,
//! i5-8600K, 16 GB, growable 24 GB paging file; RF-DETR nano 384, batch 8, 2 epochs, workers
//! forced; docs/TRAINING.md "Data loader workers"): everything else committed ~17.9 GB and 7.6 GB
//! of RAM was available before a run; the trainer 8.0 GB commit, 2.6 GB working set; per worker
//! process ~1.43 GB commit, ~0.75 GB working set.
//! - 0 workers: 8.6 img/s, data wait 35%;
//! - 2 workers: 11.4 img/s, data wait 15%, commit 31.4 of 32.4 GB, 2.06 GB of RAM left;
//! - 4 workers: no faster than 2 (11.3), then out of host memory (commit 36.6 of 37.8 GB, 0.87 GB
//!   of RAM left).

use super::*;
use crate::training::recipe::TrainRecipe;
use crate::training::service::RunRecord;

fn gb(g: f64) -> u64 {
    (g * f64::from(1u32 << 30)) as u64
}

fn dev_box() -> Machine {
    Machine {
        cpu: "Intel(R) Core(TM) i5-8600K CPU @ 3.60GHz".into(),
        cores: 6,
        ram_gb: 16,
        gpu: "NVIDIA GeForce GTX 1070 Ti".into(),
        vram_mb: 8192,
    }
}

fn nano(batch: u32) -> TrainRecipe {
    let mut r = TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap();
    r.imgsz = 384;
    r.batch = batch;
    r
}

/// A done run of `recipe` created at `created` with `res`.
fn record(created: &str, recipe: &TrainRecipe, res: RunResources) -> RunRecord {
    let mut r: RunRecord = serde_json::from_value(serde_json::json!({
        "id": created, "name": "x", "created": created,
        "request": {"recipe": recipe, "datasets": []},
        "device": "cuda", "state": "done", "stage": null, "step": 8, "steps": 8, "percent": 100,
        "detail": "", "run_dir": "/tmp/x"
    }))
    .unwrap();
    r.resources = Some(res);
    r
}

/// The dev box's run with `workers` (RF-DETR: 2 loaders).
fn measured(workers: u32) -> RunResources {
    let base = RunResources {
        workers,
        loaders: 2,
        forced: true,
        machine: dev_box(),
        commit_start: Some(gb(17.9)),
        ram_start: Some(gb(7.6)),
        samples: 40,
        ..RunResources::default()
    };
    match workers {
        0 => RunResources {
            img_s: Some(8.6),
            data_wait: Some(0.35),
            commit_peak: Some(gb(25.9)),
            commit_limit_at_peak: Some(gb(32.4)),
            ram_min: Some(gb(5.0)),
            ..base
        },
        2 => RunResources {
            img_s: Some(11.4),
            data_wait: Some(0.15),
            commit_peak: Some(gb(31.4)),
            commit_limit_at_peak: Some(gb(32.4)),
            ram_min: Some(gb(2.06)),
            page_ins_peak: Some(2800.0),
            ..base
        },
        4 => RunResources {
            img_s: Some(11.3),
            data_wait: None,
            commit_peak: Some(gb(36.6)),
            commit_limit_at_peak: Some(gb(37.8)),
            ram_min: Some(gb(0.87)),
            page_ins_peak: Some(12_000.0),
            host_oom: Some(4),
            ..base
        },
        _ => unreachable!(),
    }
}

/// The three dev box runs in the order they ran.
fn dev_box_runs() -> Vec<RunRecord> {
    [0, 2, 4]
        .iter()
        .enumerate()
        .map(|(i, &w)| record(&format!("2026-11-1{i}T10:00:00Z"), &nano(8), measured(w)))
        .collect()
}

fn learn(past: &[RunRecord], limit: u32) -> (u32, String) {
    learned(&dev_box(), &WorkerKey::of(&nano(8)), past, limit)
}

#[test]
fn no_history_gives_the_limit() {
    assert_eq!(
        learn(&[], 4),
        (4, "nothing learned on this machine yet".to_string())
    );
    // A record from before resources were recorded teaches nothing.
    let mut old = record("2026-11-01T00:00:00Z", &nano(8), measured(2));
    old.resources = None;
    assert_eq!(learn(&[old], 4).0, 4);
}

#[test]
fn the_dev_box_settles_at_two() {
    let (n, why) = learn(&dev_box_runs(), 4);
    assert_eq!(n, 2, "{why}");
    assert_eq!(
        sentence(n, &why, 4),
        "data loader workers: 2 (learned on this machine: 4 was no faster than 2; limits allow 4)"
    );
    // In any order: the records are sorted by their creation time.
    let mut shuffled = dev_box_runs();
    shuffled.reverse();
    assert_eq!(learn(&shuffled, 4), (n, why));
}

#[test]
fn the_dev_box_run_by_run() {
    let runs = dev_box_runs();
    // After 0 workers: it waited on data 35% of the time and left memory for one more worker
    // (no worker share measured: the 1.4 / 0.75 GB per loader of the limit's rule).
    let (n, why) = learn(&runs[..1], 4);
    assert_eq!(n, 1, "{why}");
    assert!(
        why.ends_with("0 waited on data 35% of the time and left memory for one more"),
        "{why}"
    );
    // After 2: it waited 15%, but its least available RAM (2.06 GB) less 2 GB does not cover one
    // more worker's measured share ((7.6 - 2.06) / 4 processes x 2 loaders), nor does its commit.
    let (n, why) = learn(&runs[..2], 4);
    assert_eq!(n, 2, "{why}");
    assert!(
        why.ends_with("2 waited on data 15% of the time, but left too little memory for one more"),
        "{why}"
    );
    assert!(!one_more_fits(&measured(2)));
    assert!(one_more_fits(&measured(0)));
    // The measured share of the 2-worker run: its growth over 4 processes, times 2 loaders.
    let (commit, ram) = measured(2).worker_share();
    assert_eq!(commit, Some((gb(31.4) - gb(17.9)) / 4 * 2));
    assert_eq!(ram, Some((gb(7.6) - gb(2.06)) / 4 * 2));
    assert_eq!(RunResources::default().worker_share(), (None, None));
}

#[test]
fn a_host_oom_caps_the_count_below_it() {
    // Only the run that ran out of memory: at most 3, whatever the limit.
    let runs = vec![record("2026-11-12T10:00:00Z", &nano(8), measured(4))];
    let (n, why) = learn(&runs, 6);
    assert_eq!(n, 3);
    assert_eq!(
        why,
        "learned on this machine: 4 ran out of system memory, so at most 3"
    );
    // Faster at 4 than at 2 but out of memory: 3 is the most, tried only when 2 waited on data
    // and left room (here it did not).
    let mut fast = measured(4);
    fast.img_s = Some(13.0);
    let runs = vec![
        record("2026-11-11T10:00:00Z", &nano(8), measured(2)),
        record("2026-11-12T10:00:00Z", &nano(8), fast),
    ];
    let (n, why) = learn(&runs, 4);
    assert_eq!(n, 2, "{why}");
    assert!(
        why.ends_with("; 4 ran out of system memory, so at most 3"),
        "{why}"
    );
    // The same with room: one more, up to the cap.
    let mut roomy = measured(2);
    roomy.ram_min = Some(gb(6.0));
    roomy.commit_limit_at_peak = Some(gb(48.0));
    let runs = vec![
        record("2026-11-11T10:00:00Z", &nano(8), roomy),
        record("2026-11-12T10:00:00Z", &nano(8), {
            let mut f = measured(4);
            f.img_s = Some(13.0);
            f
        }),
    ];
    assert_eq!(learn(&runs, 4).0, 3);
}

#[test]
fn a_restarted_run_keeps_the_failed_attempts_speed() {
    // 4 workers ran an epoch at 11.3 img/s, ran out of host memory, and the restart trained with 2.
    let res = RunResources {
        host_oom: Some(4),
        host_oom_img_s: Some(11.3),
        ..measured(2)
    };
    let runs = vec![
        record("2026-11-10T10:00:00Z", &nano(8), measured(0)),
        record("2026-11-11T10:00:00Z", &nano(8), res),
    ];
    assert_eq!(
        learn(&runs, 4),
        (
            2,
            "learned on this machine: 4 was no faster than 2".to_string()
        )
    );
}

#[test]
fn another_machine_starts_over() {
    // The host OOM of a box with less RAM caps nothing here, nor do its speeds count.
    let mut other = measured(4);
    other.machine.ram_gb = 8;
    let runs = vec![record("2026-11-12T10:00:00Z", &nano(8), other)];
    assert_eq!(learn(&runs, 4).0, 4);
    // Another GPU in the same box: the same.
    let mut gpu = measured(4);
    gpu.machine.gpu = "NVIDIA GeForce RTX 3060".into();
    assert_eq!(
        learn(&[record("2026-11-12T10:00:00Z", &nano(8), gpu)], 4).0,
        4
    );
}

#[test]
fn another_recipe_key_learns_nothing() {
    // Batch 4, input 512, another size or family: each its own key.
    let runs: Vec<RunRecord> = dev_box_runs()
        .into_iter()
        .map(|mut r| {
            r.request.recipe.batch = 4;
            r
        })
        .collect();
    assert_eq!(learn(&runs, 4).0, 4);
    let mut big = nano(8);
    big.imgsz = 512;
    assert_eq!(
        learned(&dev_box(), &WorkerKey::of(&big), &dev_box_runs(), 4).0,
        4
    );
    let small = {
        let mut r = TrainRecipe::new(TrainFamily::Rfdetr, "small").unwrap();
        r.imgsz = 384;
        r.batch = 8;
        r
    };
    assert_eq!(
        learned(&dev_box(), &WorkerKey::of(&small), &dev_box_runs(), 4).0,
        4
    );
}

#[test]
fn never_above_the_limit_of_the_moment() {
    // Learned 2, but the memory free now allows 1.
    let (n, why) = learn(&dev_box_runs(), 1);
    assert_eq!(n, 1);
    assert_eq!(
        sentence(n, &why, 1),
        "data loader workers: 1 (learned on this machine: 4 was no faster than 2; limits allow 1)"
    );
    // One more would be 1, the limit is 0.
    assert_eq!(learn(&dev_box_runs()[..1], 0).0, 0);
}

#[test]
fn more_workers_must_be_five_percent_faster() {
    let at = |w: u32, img_s: f64, created: &str| {
        let mut r = measured(2);
        r.workers = w;
        r.img_s = Some(img_s);
        r.host_oom = None;
        record(created, &nano(8), r)
    };
    // 4% faster: not worth it.
    let runs = vec![
        at(2, 10.0, "2026-11-10T10:00:00Z"),
        at(4, 10.4, "2026-11-11T10:00:00Z"),
    ];
    assert_eq!(learn(&runs, 8).0, 2);
    // 6% faster: 4.
    let runs = vec![
        at(2, 10.0, "2026-11-10T10:00:00Z"),
        at(4, 10.6, "2026-11-11T10:00:00Z"),
    ];
    let (n, why) = learn(&runs, 8);
    assert_eq!(n, 4, "{why}");
    // Each step is judged against the best below it: 1 is no faster than 0, 3 is.
    let runs = vec![
        at(0, 10.0, "2026-11-10T10:00:00Z"),
        at(1, 10.2, "2026-11-11T10:00:00Z"),
        at(3, 11.0, "2026-11-12T10:00:00Z"),
        at(4, 11.1, "2026-11-13T10:00:00Z"),
        at(5, 11.2, "2026-11-14T10:00:00Z"),
    ];
    let (n, why) = learn(&runs, 8);
    assert_eq!(n, 3, "{why}");
    assert!(why.ends_with("4 and 5 were no faster than 3"), "{why}");
    // The latest run of a count is its speed.
    let runs = vec![
        at(2, 10.0, "2026-11-10T10:00:00Z"),
        at(4, 12.0, "2026-11-11T10:00:00Z"),
        at(4, 10.1, "2026-11-12T10:00:00Z"),
    ];
    assert_eq!(learn(&runs, 8).0, 2);
}

#[test]
fn a_run_that_did_not_wait_on_data_stays() {
    let mut r = measured(2);
    r.data_wait = Some(0.05);
    r.ram_min = Some(gb(6.0));
    r.commit_limit_at_peak = Some(gb(64.0));
    let runs = vec![record("2026-11-10T10:00:00Z", &nano(8), r.clone())];
    assert_eq!(
        learn(&runs, 4),
        (
            2,
            "learned on this machine: 2 waited on data only 5% of the time".to_string()
        )
    );
    // Waiting more than 10% with room: 3.
    r.data_wait = Some(0.2);
    let runs = vec![record("2026-11-10T10:00:00Z", &nano(8), r.clone())];
    assert_eq!(learn(&runs, 4).0, 3);
    // No data wait reported (an older trainer): stays.
    r.data_wait = None;
    let runs = vec![record("2026-11-10T10:00:00Z", &nano(8), r)];
    assert_eq!(learn(&runs, 4).0, 2);
}

#[test]
fn forced_runs_count_and_unmeasured_runs_do_not() {
    // Forced (training.workers) or automatic, a measurement is a measurement.
    let mut auto = dev_box_runs();
    for r in &mut auto {
        let x = r.resources.as_mut().unwrap();
        x.forced = false;
        x.limit = Some(4);
    }
    assert_eq!(learn(&auto, 4).0, 2);
    // A run without a finished epoch and no memory failure: nothing measured.
    let mut r = measured(2);
    r.img_s = None;
    let (n, why) = learn(&[record("2026-11-10T10:00:00Z", &nano(8), r)], 4);
    assert_eq!(n, 4);
    assert!(why.starts_with("nothing measured"), "{why}");
}

#[test]
fn a_figure_the_os_does_not_give_is_not_checked() {
    // Linux: no commit limit; RAM read.
    let r = RunResources {
        commit_start: None,
        commit_peak: None,
        commit_limit_at_peak: None,
        ram_min: Some(gb(6.0)),
        ..measured(2)
    };
    assert!(one_more_fits(&r));
    // Nothing read at all (macOS): only the data wait decides.
    let bare = RunResources {
        workers: 2,
        loaders: 2,
        ..RunResources::default()
    };
    assert!(one_more_fits(&bare));
}

#[test]
fn old_records_load_and_resources_round_trip() {
    // A record without the field (every record before it) loads with None.
    let r = record("2026-11-10T10:00:00Z", &nano(8), measured(2));
    let mut v = serde_json::to_value(&r).unwrap();
    assert!(v.get("resources").is_some());
    v.as_object_mut().unwrap().remove("resources");
    let back: RunRecord = serde_json::from_value(v.clone()).unwrap();
    assert_eq!(back.resources, None);
    // And None is not written.
    assert!(
        serde_json::to_value(&back)
            .unwrap()
            .get("resources")
            .is_none()
    );
    // The field round-trips, and a partial one (a later version's extra keys, fewer keys) loads.
    let back: RunRecord = serde_json::from_value(serde_json::to_value(&r).unwrap()).unwrap();
    assert_eq!(back.resources, r.resources);
    v["resources"] = serde_json::json!({"workers": 3, "future": 1});
    let back: RunRecord = serde_json::from_value(v).unwrap();
    assert_eq!(back.resources.map(|x| x.workers), Some(3));
}

#[test]
fn medians_and_the_resource_line() {
    use crate::training::progress::EpochPoint;
    let ep = |e: u32, img_s: Option<f64>| EpochPoint {
        epoch: e,
        elapsed_s: 1.0,
        metrics: img_s
            .map(|v| ("img_s".to_string(), v))
            .into_iter()
            .collect(),
    };
    assert_eq!(median_metric(&[], "img_s"), None);
    assert_eq!(
        median_metric(
            &[ep(1, Some(8.0)), ep(2, None), ep(3, Some(f64::NAN))],
            "img_s"
        ),
        Some(8.0)
    );
    assert_eq!(
        median_metric(
            &[ep(1, Some(3.0)), ep(2, Some(1.0)), ep(3, Some(2.0))],
            "img_s"
        ),
        Some(2.0)
    );
    assert_eq!(
        median_metric(&[ep(1, Some(4.0)), ep(2, Some(1.0))], "img_s"),
        Some(2.5)
    );
    assert_eq!(
        measured(2).describe(),
        "resources: 2 data loader workers (4 processes), commit 17.9 GB at the start, peak 31.4 \
         of 32.4 GB, RAM 7.6 GB free at the start, 2.06 GB at least, page-ins up to 2.8k/s, 11.4 \
         img/s, waited on data 15%"
    );
    assert!(measured(4).describe().ends_with(
        "page-ins up to 12k/s, 11.3 img/s, out of system memory with 4 data loader workers"
    ));
    assert_eq!(
        RunResources::default().describe(),
        "resources: 0 data loader workers (0 processes)"
    );
}

#[test]
fn learned_now_lists_each_key_of_this_machine() {
    let accel = crate::training::capability::Accelerator::Mps {
        name: "Test GPU".into(),
        memory_mb: 8192,
    };
    let here = Machine::now(&accel);
    let mut runs = dev_box_runs();
    for r in &mut runs {
        r.resources.as_mut().unwrap().machine = here.clone();
    }
    let mut other = runs[1].clone();
    other.request.recipe.batch = 4;
    other.created = "2026-11-19T00:00:00Z".into();
    runs.push(other);
    let v = learned_now(&accel, None, &runs);
    assert_eq!(v.len(), 2, "{v:?}");
    assert_eq!(v[0]["batch"], 8);
    assert_eq!(v[1]["batch"], 4);
    let limit = v[0]["limit"].as_u64().unwrap() as u32;
    assert_eq!(v[0]["learned"].as_u64().unwrap() as u32, 2.min(limit));
    assert!(
        v[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("data loader workers: ")
    );
    let forced = learned_now(&accel, Some(1), &runs);
    assert_eq!(forced[0]["workers"], 1);
    assert_eq!(
        forced[0]["text"],
        "data loader workers: 1 (training.workers)"
    );
    // Runs of another machine are not listed.
    if here != dev_box() {
        assert!(learned_now(&accel, None, &dev_box_runs()).is_empty());
    }
}
