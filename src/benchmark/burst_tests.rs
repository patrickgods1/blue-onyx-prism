//! The burst test ([`super`]) on the fake backend: open-loop timing against a hand computation,
//! refusals by reason, cancel mid-burst, teardown (thread joined, backend dropped), no third copy
//! of the model, a worker that cannot load; and the pure parts (schedule, summary, frame pool).
//! Timing assertions use the fake's scripted inference delays with generous upper tolerances
//! (CI runners stall); lower bounds come from the scripted delays, which the fake never beats.

use super::*;
use crate::backend::fake::{FakeScript, Step};
use crate::backend::spec::Device;
use crate::benchmark::images::{GroundTruth, SetImage, SetKind};
use crate::benchmark::{Bench, Job, is_cancelled};
use crate::model::{ModelFamilyKind, PostParams};
use std::path::{Path, PathBuf};

/// A temp folder removed on drop.
struct Dir(PathBuf);

impl Dir {
    fn new() -> Self {
        let d = std::env::temp_dir().join(format!("bop-burst-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Self(d)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Model input of the fake.
const INPUT: (u32, u32) = (64, 64);
/// Generous bound for what should take well under a second (CI runners stall for seconds).
const SLACK_MS: f64 = 600.0;

fn jpeg(dir: &Path, name: &str, seed: u8) -> SetImage {
    let img = crate::image::RgbImage {
        width: 64,
        height: 48,
        rgb: (0..64u32 * 48 * 3)
            .map(|i| (i as u8).wrapping_mul(seed))
            .collect(),
    };
    let path = dir.join(name);
    std::fs::write(&path, crate::image::encode_jpeg(&img, 90).unwrap()).unwrap();
    SetImage {
        file: name.into(),
        path: Some(path),
        width: 64,
        height: 48,
        tags: Vec::new(),
        gt: None,
    }
}

fn set(id: &str, images: Vec<SetImage>) -> ImageSet {
    ImageSet {
        id: id.into(),
        title: id.into(),
        kind: SetKind::Dir,
        dir: None,
        license: String::new(),
        attribution: String::new(),
        source: String::new(),
        ground_truth: GroundTruth::None,
        scored_labels: None,
        transform: None,
        images,
    }
}

fn sets(dir: &Path) -> Vec<ImageSet> {
    vec![set(
        "day",
        (0..3)
            .map(|i| jpeg(dir, &format!("f{i}.jpg"), i + 3))
            .collect(),
    )]
}

fn job(dir: &Path) -> Job {
    Job {
        name: "fake".into(),
        path: dir.join("fake.onnx"),
        family: ModelFamilyKind::Yolo26,
        classes: vec!["person".into(), "car".into()],
        device: "openvino:cpu".into(),
        gpu_precision: None,
        confidence_threshold: None,
        object_filter: None,
        variant: None,
        concurrency: None,
        device_thresholds: Default::default(),
    }
}

fn cpu() -> Device {
    *spec::parse("openvino:cpu").unwrap().device().unwrap()
}

/// One decode + preprocess thread and the given worker queue (0 = auto-sized).
fn config(queue: usize) -> Config {
    Config {
        preprocess_threads: 1,
        worker_queue_size: queue,
        ..Config::default()
    }
}

fn load(cameras: usize, images: usize, interval_ms: u64) -> BurstConfig {
    BurstConfig {
        enabled: true,
        cameras,
        images,
        interval_ms,
    }
}

const PARAMS: PostParams = PostParams {
    confidence_threshold: 0.5,
    nms_iou: 0.45,
};

/// `n` burst inferences of `delay_ms` each.
fn slow(script: &FakeScript, n: usize, delay_ms: u64) {
    script.push((0..n).map(|_| {
        Step::detections(
            &[[10.0, 10.0, 30.0, 30.0, 0.9, 0.0]],
            Duration::from_millis(delay_ms),
        )
    }));
}

/// A bench over `sets` without warm-up, one timed run per image.
fn bench(sets: &[ImageSet]) -> Bench<'_> {
    let mut b = Bench::new(sets, 0, 1, PARAMS);
    b.warmup_rule = crate::benchmark::WarmupRule::COUNT_ONLY;
    b
}

fn burst<'a>(
    settings: BurstConfig,
    rt: &'a Arc<Mutex<Runtimes>>,
    config: &'a Config,
    bursts: usize,
) -> Burst<'a> {
    Burst {
        bursts,
        settle: Duration::from_millis(50),
        load_timeout: Duration::from_secs(30),
        ..Burst::new(settings, rt, config)
    }
}

/// Sorted request latencies a hand computation gives: one slot, FIFO, `service` ms per
/// inference, every camera's k-th request due at k x `interval`.
fn hand_latencies(cameras: usize, images: usize, interval: f64, service: f64) -> Vec<f64> {
    let mut free = 0.0f64;
    let mut out = Vec::new();
    for k in 0..images {
        let due = k as f64 * interval;
        for _ in 0..cameras {
            let start = free.max(due);
            free = start + service;
            out.push(free - due);
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

/// Open loop on one slot: later requests wait in the queue, and their latency (from the
/// intended send time) is the hand computation's: 2 cameras x 3 images 100 ms apart on a model
/// of 150 ms give 150, 300, 350, 500, 550, 700 ms; the triggers last 750 and 900 ms.
#[test]
fn open_loop_latency_includes_the_queue_wait_from_the_intended_time() {
    let dir = Dir::new();
    let script = FakeScript::new(1, INPUT);
    let rt = Arc::new(Mutex::new(Runtimes::fake(script.clone())));
    let cfg = config(0);
    let s = sets(&dir.0);
    slow(&script, 6, 150);
    let b = burst(load(2, 3, 100), &rt, &cfg, 1);
    let mut bench = bench(&s);
    bench.burst = Some(&b);
    let r = bench.run_exact(&rt, &job(&dir.0), cpu()).unwrap();
    let br = r.burst.expect("a burst result");
    assert_eq!(br.error, None, "{:?}", br.notes);
    assert_eq!((br.requests, br.answered, br.triggers), (6, 6, 2));
    assert!(br.refused.is_empty(), "{:?}", br.refused);
    let hand = hand_latencies(2, 3, 100.0, 150.0);
    assert_eq!(hand, vec![150.0, 300.0, 350.0, 500.0, 550.0, 700.0]);
    // Nearest rank over the six: p50 = the 3rd, p95 = p99 = max = the 6th.
    let l = &br.latency_ms;
    for (got, want) in [
        (l.min, 150.0),
        (l.p50, 350.0),
        (l.p95, 700.0),
        (l.max, 700.0),
    ] {
        assert!(
            got >= want - 5.0 && got <= want + SLACK_MS,
            "latency {got} ms, hand {want} ms ({l:?})"
        );
    }
    let t = &br.trigger_ms;
    assert!(
        t.min >= 745.0 && t.min <= 750.0 + SLACK_MS,
        "first trigger {t:?}"
    );
    assert!(
        t.max >= 895.0 && t.max <= 900.0 + SLACK_MS,
        "last trigger {t:?}"
    );
    assert!(
        br.max_queue_depth >= 2,
        "two requests were waiting at once: {}",
        br.max_queue_depth
    );
    assert_eq!(br.frames, 3);
    assert_eq!(br.frame_sets, vec!["day".to_string()]);
    assert!(!br.served);
    assert_eq!(script.submits(), 6, "every burst request inferred once");
}

/// A late send still counts from when it was due (no coordinated omission); refusals are the
/// drop counters by reason plus `error` for the rest; a trigger with a refused request is
/// incomplete and stays out of the trigger times.
#[test]
fn the_summary_counts_from_the_intended_time() {
    let ms = Duration::from_millis;
    let due = |camera, image, at| Due {
        camera,
        image,
        at: ms(at),
    };
    let o = |burst, d: Due, sent, answered, ok: bool| Outcome {
        burst,
        due: d,
        sent: ms(sent),
        answered: ms(answered),
        ok,
        error: (!ok).then(|| "Worker queue is full".to_string()),
    };
    let outcomes = vec![
        // Sent 400 ms late, answered 100 ms after its send: 500 ms from when it was due.
        o(0, due(0, 0, 0), 400, 500, true),
        o(0, due(0, 1, 100), 400, 600, true),
        o(0, due(1, 0, 0), 0, 50, true),
        o(0, due(1, 1, 100), 100, 101, false),
        o(1, due(0, 0, 0), 0, 30, false),
    ];
    let mut r = BurstResult::default();
    summarize(
        &mut r,
        &outcomes,
        &[(DropReason::QueueFull, 1), (DropReason::Timeout, 0)],
    );
    assert_eq!((r.requests, r.answered), (5, 3));
    assert_eq!(r.latency_ms.max, 500.0);
    assert_eq!(r.latency_ms.min, 50.0);
    assert_eq!(r.latency_ms.p50, 500.0, "sorted 50, 500, 500");
    assert_eq!(r.max_send_lag_ms, 400.0);
    // Triggers: (0, cam0) complete 0..600; (0, cam1) and (1, cam0) incomplete.
    assert_eq!((r.triggers, r.triggers_incomplete), (3, 2));
    assert_eq!(r.trigger_ms.max, 600.0);
    let want: BTreeMap<String, u64> = [("queue_full".to_string(), 1), ("error".to_string(), 1)]
        .into_iter()
        .collect();
    assert_eq!(r.refused, want);
    assert_eq!(r.refused_total(), 2);
    assert_eq!(r.first_refusal.as_deref(), Some("Worker queue is full"));
}

/// A full queue refuses requests as it refuses Blue Iris' (`queue_full`); every request is
/// either answered or refused.
#[test]
fn a_full_queue_refuses_and_is_counted_by_reason() {
    let dir = Dir::new();
    let script = FakeScript::new(1, INPUT);
    let rt = Arc::new(Mutex::new(Runtimes::fake(script.clone())));
    // A queue of one request: 8 cameras at once overflow it.
    let cfg = config(1);
    let s = sets(&dir.0);
    slow(&script, 8, 300);
    let b = burst(load(8, 1, 100), &rt, &cfg, 1);
    let mut bench = bench(&s);
    bench.burst = Some(&b);
    let br = bench
        .run_exact(&rt, &job(&dir.0), cpu())
        .unwrap()
        .burst
        .unwrap();
    assert_eq!(br.error, None);
    assert_eq!(br.requests, 8);
    assert_eq!(br.queue_capacity, 1);
    let full = br.refused.get("queue_full").copied().unwrap_or(0);
    assert!(
        (3..=7).contains(&full),
        "at most the queue, one waiting for a slot and one inferring get in: {:?}",
        br.refused
    );
    assert_eq!(br.answered as u64 + br.refused_total(), 8);
    assert_eq!(br.refused.len(), 1, "{:?}", br.refused);
    assert!(
        br.first_refusal
            .as_deref()
            .unwrap()
            .contains("queue is full"),
        "{:?}",
        br.first_refusal
    );
    assert_eq!(br.triggers_incomplete, full as usize);
}

/// A cancel during a burst ends it promptly: the run fails with `Cancelled`, the worker is torn
/// down (its backend dropped) long before the scripted burst would have ended.
#[test]
fn a_cancel_mid_burst_stops_it_and_tears_the_worker_down() {
    let dir = Dir::new();
    let script = FakeScript::new(1, INPUT);
    let rt = Arc::new(Mutex::new(Runtimes::fake(script.clone())));
    let cfg = config(0);
    let s = sets(&dir.0);
    // 20 requests of 500 ms on one slot: 10 s uncancelled.
    slow(&script, 20, 500);
    let b = burst(load(4, 5, 200), &rt, &cfg, 3);
    let cancel = AtomicBool::new(false);
    let cancelled_at = Mutex::new(None::<Instant>);
    let progress = |p: &crate::benchmark::Progress| {
        if p.phase == Phase::Burst && p.done >= 1 && !cancel.swap(true, Ordering::SeqCst) {
            *cancelled_at.lock().unwrap() = Some(Instant::now());
        }
    };
    let mut bench = bench(&s);
    bench.burst = Some(&b);
    bench.cancel = Some(&cancel);
    bench.progress = Some(&progress);
    let e = bench.run_exact(&rt, &job(&dir.0), cpu()).unwrap_err();
    let after = cancelled_at.lock().unwrap().expect("cancelled").elapsed();
    assert!(is_cancelled(&e), "{e:#}");
    assert!(
        after < Duration::from_secs(5),
        "stopped {after:?} after the cancel"
    );
    assert_eq!(script.alive(), 0, "the burst's backend was dropped");
    assert!(script.submits() < 20, "the burst did not run to its end");
}

/// Teardown and copies: the benchmark's own compiled copy is dropped before the burst's worker
/// compiles (with a serving copy alive the whole time, never more than two at once: serving +
/// one), and after the run every copy the run made is dropped: the worker's thread ended.
#[test]
fn no_third_copy_and_everything_is_dropped_after() {
    let dir = Dir::new();
    let script = FakeScript::new(2, INPUT);
    let rt = Arc::new(Mutex::new(Runtimes::fake(script.clone())));
    // The serving worker's copy of the model, alive through the whole run.
    let serving = script.compile(
        &LoadRequest {
            path: dir.0.join("fake.onnx"),
            requested: "openvino:cpu".into(),
            gpu_precision: None,
            no_cache: false,
        },
        "openvino:cpu",
        false,
        &crate::profile::DeviceTuning::SERIAL,
    );
    assert_eq!(script.alive(), 1);
    let cfg = config(0);
    let s = sets(&dir.0);
    slow(&script, 4, 20);
    let b = burst(load(2, 2, 50), &rt, &cfg, 2);
    let mut bench = bench(&s);
    bench.burst = Some(&b);
    let r = bench.run_exact(&rt, &job(&dir.0), cpu()).unwrap();
    let br = r.burst.unwrap();
    assert_eq!((br.requests, br.answered), (8, 8), "{:?}", br.refused);
    assert_eq!(script.compiles().len(), 3, "serving, timed run, burst");
    assert_eq!(
        script.max_alive(),
        2,
        "the serving copy and one benchmark copy at most"
    );
    assert_eq!(script.alive(), 1, "only the serving copy is left");
    assert_eq!(
        script.dropped(),
        2,
        "the timed run's and the burst's copies"
    );
    drop(serving);
    assert_eq!(script.alive(), 0);
    // Two bursts: 2 x 2 requests each, 50 ms apart.
    assert_eq!(br.bursts, 2);
    assert!(br.wall_ms >= 50.0 + 50.0, "{}", br.wall_ms);
}

/// A worker that cannot load is the result's error, and the timed run's result stands.
#[test]
fn a_worker_that_does_not_load_is_recorded_and_the_run_goes_on() {
    let dir = Dir::new();
    let script = FakeScript::new(1, INPUT);
    let rt = Arc::new(Mutex::new(Runtimes::fake(script.clone())));
    let cfg = config(0);
    let s = sets(&dir.0);
    let mut b = burst(load(2, 2, 50), &rt, &cfg, 1);
    b.faults.load = Some(Arc::new(|_c: &Candidate| {
        anyhow::bail!("out of device memory (fake)")
    }));
    let mut bench = bench(&s);
    bench.burst = Some(&b);
    let r = bench.run_exact(&rt, &job(&dir.0), cpu()).unwrap();
    assert_eq!(r.images, 3, "the timed run stands");
    let br = r.burst.unwrap();
    let e = br.error.as_deref().unwrap();
    assert!(
        e.contains("did not load") && e.contains("out of device memory"),
        "{e}"
    );
    assert_eq!(br.requests, 0);
    assert_eq!(script.alive(), 0);
    assert!(br.describe().contains("not measured"), "{}", br.describe());
}

/// Every camera's k-th request is due at k x interval, in send order.
#[test]
fn the_schedule_is_open_loop() {
    let d = schedule(&load(3, 2, 250));
    let got: Vec<(usize, usize, u128)> = d
        .iter()
        .map(|x| (x.camera, x.image, x.at.as_millis()))
        .collect();
    assert_eq!(
        got,
        vec![
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (0, 1, 250),
            (1, 1, 250),
            (2, 1, 250)
        ]
    );
}

/// Your cameras' frames come first, the pool is capped and spread over each group.
#[test]
fn the_frame_pool_takes_your_cameras_first() {
    let dir = Dir::new();
    let cams = set(
        crate::benchmark::cameras::ID,
        vec![jpeg(&dir.0, "c0.jpg", 1), jpeg(&dir.0, "c1.jpg", 2)],
    );
    let many: Vec<SetImage> = (0..100)
        .map(|i| jpeg(&dir.0, &format!("p{i}.jpg"), 3))
        .collect();
    let public = set("coco-cctv", many);
    let (pool, skipped) = frame_pool(&[public.clone(), cams.clone()]);
    assert_eq!(skipped, 0);
    assert_eq!(pool.len(), MAX_FRAMES);
    assert_eq!(
        pool.iter()
            .take(2)
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        vec!["c0.jpg", "c1.jpg"]
    );
    assert!(pool.iter().skip(2).all(|f| f.set == "coco-cctv"));
    // Spread: the first and a late public frame are both in.
    assert!(pool.iter().any(|f| f.name == "p0.jpg"));
    assert!(
        pool.iter()
            .any(|f| f.name == "p96.jpg" || f.name == "p98.jpg")
    );
    // A missing file is skipped, not fatal.
    let mut gone = cams;
    gone.images[0].path = Some(dir.0.join("missing.jpg"));
    let (pool, skipped) = frame_pool(&[gone]);
    assert_eq!((pool.len(), skipped), (1, 1));
}

/// The one-line text names the load, the percentiles and the refusals by reason.
#[test]
fn the_result_text_names_the_load_and_the_refusals() {
    let mut r = BurstResult {
        cameras: 8,
        images: 4,
        interval_ms: 500,
        requests: 96,
        answered: 94,
        ..BurstResult::default()
    };
    r.latency_ms.p95 = 480.0;
    r.trigger_ms.p95 = 1600.0;
    r.refused.insert("queue_full".into(), 2);
    let t = r.describe();
    assert!(t.starts_with("burst 8 x 4 x 500 ms"), "{t}");
    assert!(t.contains("p95 480 ms") && t.contains("p95 1.6 s"), "{t}");
    assert!(t.contains("2 refused (queue_full 2)"), "{t}");
}
