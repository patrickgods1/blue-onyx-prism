//! Read ahead (`run.rs`): the pipelined timed loop gives bit-identical predictions, the same
//! per-image results in the same order, the same `on_image` order, the same errors and the same
//! cancel behaviour as the serial loop. Driven through the fake backend (`backend::fake`) whose
//! synchronous inference is a function of each frame's pixels, over real JPEG and PNG files.

use super::*;
use crate::backend::fake::{FakeScript, InferFn, yolo26_output};
use crate::benchmark::images::{ImageTransform, SetImage, SetKind};
use std::sync::Arc;

/// A temp folder removed on drop.
struct Dir(PathBuf);

impl Dir {
    fn new() -> Self {
        let d = std::env::temp_dir().join(format!("bop-read-ahead-{}", uuid::Uuid::new_v4()));
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
const INPUT: (u32, u32) = (64, 48);

/// Detections that depend on every pixel of the input tensor: eight bands of the CHW tensor,
/// each band's mean placing a box and setting its score (some below the production threshold,
/// some below the evaluation one).
fn infer_fn() -> InferFn {
    InferFn(Arc::new(|chw: &[f32]| {
        let n = chw.len();
        let rows: Vec<[f32; 6]> = (0..8)
            .map(|k| {
                let band = chw.get(k * n / 8..(k + 1) * n / 8).unwrap_or(&[]);
                let mean = band.iter().sum::<f32>() / band.len().max(1) as f32;
                let kf = k as f32;
                let x1 = mean * 40.0 + kf;
                let y1 = (1.0 - mean) * 25.0 + kf * 0.5;
                let score = 0.02 + 0.96 * (mean * 7.3 + kf * 0.13).fract();
                [x1, y1, x1 + 8.0 + kf, y1 + 10.0, score, (k % 3) as f32]
            })
            .collect();
        vec![yolo26_output(&rows)]
    }))
}

/// A frame of `w` x `h` whose pixels depend on `seed`, saved as `name` (JPEG or PNG by its
/// extension).
fn write_image(dir: &Path, name: &str, w: u32, h: u32, seed: u32) -> SetImage {
    let img = image::RgbImage::from_fn(w, h, |x, y| {
        let v = x.wrapping_mul(31 + seed) ^ y.wrapping_mul(17 + 3 * seed);
        image::Rgb([
            (v % 251) as u8,
            ((x + seed * 13) % 256) as u8,
            ((y * 7 + seed) % 256) as u8,
        ])
    });
    let path = dir.join(name);
    img.save(&path).unwrap();
    SetImage {
        file: name.into(),
        path: Some(path),
        width: w,
        height: h,
        tags: Vec::new(),
        gt: None,
    }
}

fn set(id: &str, images: Vec<SetImage>, transform: Option<ImageTransform>) -> ImageSet {
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
        transform,
        images,
    }
}

/// Two sets (the second grayscale, the benchmark's NIR proxy) of JPEGs and PNGs in several
/// sizes.
fn sets(dir: &Path) -> Vec<ImageSet> {
    let day = vec![
        write_image(dir, "a.jpg", 640, 480, 1),
        write_image(dir, "b.png", 320, 200, 2),
        write_image(dir, "c.jpg", 1344, 760, 3),
        write_image(dir, "d.jpg", 1280, 720, 4),
        write_image(dir, "e.png", 97, 61, 5),
    ];
    let night = vec![
        write_image(dir, "f.jpg", 800, 600, 6),
        write_image(dir, "g.jpg", 960, 540, 7),
    ];
    vec![
        set("day", day, None),
        set("night", night, Some(ImageTransform::Grayscale)),
    ]
}

fn job(dir: &Path) -> Job {
    Job {
        name: "fake".into(),
        path: dir.join("fake.onnx"),
        family: ModelFamilyKind::Yolo26,
        classes: vec!["person".into(), "car".into(), "dog".into()],
        device: "openvino:cpu".into(),
        gpu_precision: None,
        confidence_threshold: None,
        object_filter: None,
        variant: None,
        concurrency: None,
        device_thresholds: Default::default(),
    }
}

fn runtimes() -> Mutex<Runtimes> {
    let mut script = FakeScript::new(1, INPUT);
    script.infer_fn = Some(infer_fn());
    Mutex::new(Runtimes::fake(script))
}

const PARAMS: PostParams = PostParams {
    confidence_threshold: 0.4,
    nms_iou: 0.45,
};

/// A prediction as exact bits.
fn bits(label: &str, confidence: f32, bbox: [f32; 4]) -> (String, u32, [u32; 4]) {
    (
        label.to_string(),
        confidence.to_bits(),
        bbox.map(f32::to_bits),
    )
}

/// What must not change with read ahead: per image (set, file, stored predictions as bits)
/// and the evaluation predictions as bits, in order.
type Outcome = (
    Vec<(String, String, Vec<(String, u32, [u32; 4])>)>,
    Vec<Vec<(String, u32, [u32; 4])>>,
);

fn outcome(r: &RunResult) -> Outcome {
    let per_image = r
        .per_image
        .iter()
        .map(|i| {
            let preds = i
                .preds
                .iter()
                .map(|p| bits(&p.label, p.confidence, p.bbox))
                .collect();
            (i.set.clone(), i.file.clone(), preds)
        })
        .collect();
    let eval = r
        .eval_preds
        .iter()
        .map(|v| {
            v.iter()
                .map(|p| bits(&p.label, p.confidence, p.bbox))
                .collect()
        })
        .collect();
    (per_image, eval)
}

/// Run `sets` with `read_ahead`; the result and the files `on_image` saw, in call order.
fn run(
    sets: &[ImageSet],
    job: &Job,
    read_ahead: ReadAhead,
    repeat: usize,
    cancel_after: Option<usize>,
) -> (Result<RunResult>, Vec<String>) {
    let rt = runtimes();
    let seen: Mutex<Vec<String>> = Mutex::default();
    let cancel = AtomicBool::new(false);
    let sink = |r: &ImageRun| {
        let mut s = seen.lock_recover();
        s.push(r.file.clone());
        if cancel_after.is_some_and(|n| s.len() >= n) {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let mut bench = Bench::new(sets, 1, repeat, PARAMS);
    bench.warmup_rule = WarmupRule::COUNT_ONLY;
    bench.read_ahead = read_ahead;
    bench.on_image = Some(&sink);
    bench.cancel = Some(&cancel);
    bench.max_stored_preds = 300;
    let r = bench.run(&rt, job, &job.device);
    (r, seen.into_inner().unwrap())
}

#[test]
fn read_ahead_matches_the_serial_loop_bit_for_bit() {
    let dir = Dir::new();
    let sets = sets(&dir.0);
    let job = job(&dir.0);
    let files: Vec<String> = sets
        .iter()
        .flat_map(|s| s.images.iter().map(|i| i.file.clone()))
        .collect();
    for repeat in [1, 3] {
        let (serial, serial_seen) = run(&sets, &job, ReadAhead::Off, repeat, None);
        let (ahead, ahead_seen) = run(&sets, &job, ReadAhead::Always, repeat, None);
        let (serial, ahead) = (serial.unwrap(), ahead.unwrap());
        assert!(!serial.pipelined && ahead.pipelined);
        assert_eq!(outcome(&serial), outcome(&ahead), "repeat {repeat}");
        // The fake's boxes reach both thresholds: the comparison is not of empty lists.
        let (per_image, eval) = outcome(&ahead);
        let stored: usize = per_image.iter().map(|i| i.2.len()).sum();
        let evaluated: usize = eval.iter().map(Vec::len).sum();
        assert!(stored > 0 && evaluated > stored, "{stored} {evaluated}");
        assert_eq!(serial_seen, files);
        assert_eq!(ahead_seen, files, "on_image fires in image order");
        assert_eq!(
            per_image.iter().map(|i| i.1.clone()).collect::<Vec<_>>(),
            files
        );
        assert_eq!(serial.detections, ahead.detections);
        // Every run timed every stage, decode included (measured on the reader thread).
        for r in [&serial, &ahead] {
            assert_eq!(r.repeat, repeat);
            assert_eq!(r.per_image.len(), files.len());
            assert!(r.stages_ms.decode.min > 0.0, "{:?}", r.stages_ms.decode);
            assert!(r.stages_ms.total.min >= r.stages_ms.decode.min);
            assert!(r.throughput_fps > 0.0);
            assert_eq!(
                r.by_resolution.iter().map(|b| b.images).sum::<usize>(),
                files.len()
            );
        }
    }
}

/// A benchmark's default reads ahead except on a CPU device (the fake loads on `openvino:cpu`).
#[test]
fn benchmarks_do_not_read_ahead_on_cpu() {
    for (device, applies) in [
        ("openvino:cpu", false),
        ("ort:cpu", false),
        ("CPU", false),
        ("openvino:gpu.0", true),
        ("ort:cuda:0", true),
        ("ort:directml", true),
        ("ort:coreml", true),
        ("openvino:npu", true),
    ] {
        assert_eq!(ReadAhead::ExceptCpu.applies(device), applies, "{device}");
        assert!(ReadAhead::Always.applies(device) && !ReadAhead::Off.applies(device));
    }
    let dir = Dir::new();
    let sets = sets(&dir.0);
    let job = job(&dir.0);
    let (default, _) = run(
        &sets,
        &job,
        Bench::new(&[], 0, 1, PARAMS).read_ahead,
        1,
        None,
    );
    let (ahead, _) = run(&sets, &job, ReadAhead::Always, 1, None);
    let (default, ahead) = (default.unwrap(), ahead.unwrap());
    assert!(!default.pipelined && ahead.pipelined);
    assert_eq!(outcome(&default), outcome(&ahead));
}

#[test]
fn read_ahead_stops_at_a_cancel_with_the_finished_images() {
    let dir = Dir::new();
    let sets = sets(&dir.0);
    let job = job(&dir.0);
    for read_ahead in [ReadAhead::Off, ReadAhead::Always] {
        let (r, seen) = run(&sets, &job, read_ahead, 2, Some(3));
        let e = r.unwrap_err();
        assert!(is_cancelled(&e), "{e:#}");
        assert_eq!(seen, ["a.jpg", "b.png", "c.jpg"], "{read_ahead:?}");
    }
}

#[test]
fn read_ahead_fails_like_the_serial_loop() {
    let dir = Dir::new();
    let mut sets = sets(&dir.0);
    let job = job(&dir.0);
    // A file that is not an image (decode error) in the middle of the first set.
    let bad = dir.0.join("bad.jpg");
    std::fs::write(&bad, b"\xFF\xD8 not a jpeg").unwrap();
    let mut broken = sets.clone();
    broken[0].images.insert(
        2,
        SetImage {
            file: "bad.jpg".into(),
            path: Some(bad),
            width: 10,
            height: 10,
            tags: Vec::new(),
            gt: None,
        },
    );
    let (serial, serial_seen) = run(&broken, &job, ReadAhead::Off, 2, None);
    let (ahead, ahead_seen) = run(&broken, &job, ReadAhead::Always, 2, None);
    let (serial, ahead) = (serial.unwrap_err(), ahead.unwrap_err());
    assert_eq!(format!("{serial:#}"), format!("{ahead:#}"));
    assert!(format!("{ahead:#}").contains("image bad.jpg"), "{ahead:#}");
    assert_eq!(serial_seen, ["a.jpg", "b.png"]);
    assert_eq!(ahead_seen, serial_seen);

    // A file that cannot be read.
    sets[1].images[0].path = Some(dir.0.join("missing.jpg"));
    let (serial, serial_seen) = run(&sets, &job, ReadAhead::Off, 1, None);
    let (ahead, ahead_seen) = run(&sets, &job, ReadAhead::Always, 1, None);
    let (serial, ahead) = (serial.unwrap_err(), ahead.unwrap_err());
    assert_eq!(format!("{serial:#}"), format!("{ahead:#}"));
    assert_eq!(serial_seen.len(), 5);
    assert_eq!(ahead_seen, serial_seen);
}

/// A PNG of `w` x `h` from `f(x, y)` saved as `name` (lossless, so the computed scene is that of
/// these exact pixels).
fn write_png(dir: &Path, name: &str, w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> SetImage {
    let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb(f(x, y)));
    let path = dir.join(name);
    img.save(&path).unwrap();
    SetImage {
        file: name.into(),
        path: Some(path),
        width: w,
        height: h,
        tags: Vec::new(),
        gt: None,
    }
}

/// The colourful frame of the `image_stats` tests (hue varies independently of brightness).
fn colour(x: u32, y: u32) -> [u8; 3] {
    [
        ((x * 7) % 256) as u8,
        ((y * 5 + 60) % 256) as u8,
        (((x + y) * 3 + 120) % 256) as u8,
    ]
}

/// The decode stage computes the scene the service does (`ImageStats::scene` after decode): a
/// colour frame is day, the same frame dark is night, and a set's grayscale transform (the NIR
/// proxy) makes it IR: the scene is that of the frame the set sends, the transform untimed.
#[test]
fn decode_computes_the_services_scene() {
    use crate::image_stats::Scene;
    let dir = Dir::new();
    let day = write_png(&dir.0, "day.png", 320, 240, colour);
    let night = write_png(&dir.0, "night.png", 320, 240, |x, y| {
        colour(x, y).map(|v| v / 8)
    });
    let bytes = |i: &SetImage| std::fs::read(i.path.as_ref().unwrap()).unwrap();
    assert_eq!(super::run::decode_scene(&bytes(&day), None), Scene::Day);
    assert_eq!(super::run::decode_scene(&bytes(&night), None), Scene::Night);
    assert_eq!(
        super::run::decode_scene(&bytes(&day), Some(ImageTransform::Grayscale)),
        Scene::Ir
    );
    // A run records each image's scene, and the scene splits follow it (not the tags).
    let mut tagged_day = night.clone();
    tagged_day.file = "tagged-day.png".into();
    tagged_day.tags = vec!["day".into()];
    let sets = vec![
        set("a", vec![day.clone(), night.clone(), tagged_day], None),
        set("nir", vec![day], Some(ImageTransform::Grayscale)),
    ];
    let job = job(&dir.0);
    let (r, _) = run(&sets, &job, ReadAhead::Off, 1, None);
    let r = r.unwrap();
    let scenes: Vec<_> = r.per_image.iter().map(|i| i.scene).collect();
    assert_eq!(
        scenes,
        [
            Some(Scene::Day),
            Some(Scene::Night),
            Some(Scene::Night),
            Some(Scene::Ir)
        ]
    );
    let in_split = |k: usize, key: &str| {
        let img = &sets.iter().flat_map(|s| &s.images).nth(k).unwrap();
        crate::benchmark::images::in_split(&img.tags, scenes[k], key)
    };
    assert!(
        in_split(2, "night") && !in_split(2, "day"),
        "computed over tagged"
    );
    assert!(crate::benchmark::images::in_split(
        &["day".to_string()],
        None,
        "day"
    ));
    assert!(!crate::benchmark::images::in_split(&[], None, "day"));
}

/// B5: the production predictions use the threshold the service uses on the device the model
/// loaded on: the model's profile for that device, else its own threshold, else the global one.
#[test]
fn predictions_use_the_served_threshold() {
    let dir = Dir::new();
    let sets = sets(&dir.0);
    let mut job = job(&dir.0);
    let min_conf = |r: &RunResult| {
        r.per_image
            .iter()
            .flat_map(|i| &i.preds)
            .map(|p| p.confidence)
            .fold(1.0f32, f32::min)
    };
    let (global, _) = run(&sets, &job, ReadAhead::Off, 1, None);
    let global = global.unwrap();
    assert_eq!(global.served_threshold, Some(PARAMS.confidence_threshold));
    assert_eq!(global.served_threshold_source.as_deref(), Some("global"));
    assert!(
        min_conf(&global) < 0.7,
        "the fake's boxes span the thresholds"
    );

    job.confidence_threshold = Some(0.5);
    let (model, _) = run(&sets, &job, ReadAhead::Off, 1, None);
    let model = model.unwrap();
    assert_eq!(
        (
            model.served_threshold,
            model.served_threshold_source.as_deref()
        ),
        (Some(0.5), Some("model"))
    );

    // A profile for the device it ran on (`openvino:cpu`) wins; one for another device does not.
    job.device_thresholds = [("openvino:gpu".to_string(), 0.2)].into();
    let (other, _) = run(&sets, &job, ReadAhead::Off, 1, None);
    assert_eq!(other.unwrap().served_threshold, Some(0.5));
    job.device_thresholds.insert("openvino:cpu".into(), 0.7);
    let (profiled, _) = run(&sets, &job, ReadAhead::Off, 1, None);
    let profiled = profiled.unwrap();
    assert_eq!(
        (
            profiled.served_threshold,
            profiled.served_threshold_source.as_deref()
        ),
        (Some(0.7), Some("profile for openvino:cpu"))
    );
    assert!(min_conf(&profiled) >= 0.7, "{}", min_conf(&profiled));
    // The evaluation predictions are unchanged (0.05+), so AP does not move with it.
    assert_eq!(outcome(&profiled).1, outcome(&global).1);
}

/// B4: the benchmark compiles with the worker's tuning ([`crate::worker::device_tuning`], also
/// `worker::run_tests::the_worker_compiles_with_the_shared_tuning`) and records it: THROUGHPUT on
/// a 16-thread CPU, `performance_hint: throughput` on an integrated GPU, serial without a
/// profile.
#[test]
fn the_benchmark_compiles_with_the_workers_tuning() {
    use crate::backend::spec::Target;
    use crate::config::{Config, ModelConfig};
    use crate::profile::{DeviceTuning, Facts, PerfHint, PerformanceHint, classify};
    let dir = Dir::new();
    let sets = sets(&dir.0);
    let hw = crate::backend::HardwareInfo::new("linux", "x86_64", vec![]);
    let profile = classify(&Facts::from_hardware(16, Some(32.0), &hw));
    let cpu = Device::new(Runtime::OpenVino, Target::Cpu, None).unwrap();
    let gpu = Device::new(Runtime::OpenVino, Target::Gpu, None).unwrap();
    for (device, hint, expected) in [
        (cpu, None, PerfHint::Throughput),
        (gpu, None, PerfHint::Latency),
        (gpu, Some(PerformanceHint::Throughput), PerfHint::Throughput),
    ] {
        let m = ModelConfig {
            path: dir.0.join("fake.onnx"),
            performance_hint: hint,
            ..ModelConfig::default()
        };
        let config = Config {
            models: vec![m.clone()],
            ..Config::default()
        };
        let mut job = job(&dir.0);
        let c = crate::worker::Concurrency::for_model(profile.clone(), &config, &m);
        job.concurrency = Some(c.clone());
        let mut script = FakeScript::new(1, INPUT);
        script.infer_fn = Some(infer_fn());
        let rt = Mutex::new(Runtimes::fake(script.clone()));
        let mut bench = Bench::new(&sets, 0, 1, PARAMS);
        bench.warmup_rule = WarmupRule::COUNT_ONLY;
        let r = bench.run_exact(&rt, &job, device).unwrap();
        let want = crate::worker::device_tuning(Some(&c), &super::run::lock(&rt), &device);
        assert_eq!(want.hint, expected, "{device} {hint:?}");
        assert_eq!(script.compiles(), vec![(device.to_string(), want)]);
        assert_eq!(
            r.tuning,
            Some(RunTuning {
                hint: Some(expected.as_str().to_string()),
                inflight: 1,
                ort_sessions: 1,
            })
        );
    }
    // No profile (detections over frames): serial.
    let mut script = FakeScript::new(1, INPUT);
    script.infer_fn = Some(infer_fn());
    let rt = Mutex::new(Runtimes::fake(script.clone()));
    let bench = Bench::new(&sets, 0, 1, PARAMS);
    bench.run_exact(&rt, &job(&dir.0), cpu).unwrap();
    assert_eq!(
        script.compiles(),
        vec![(cpu.to_string(), DeviceTuning::SERIAL)]
    );
}

/// B6: the throughput timer covers the wait for the frame and the served stages only: a slow
/// evaluation post-process (the benchmark's own untimed work) does not lower req/s.
#[test]
fn throughput_leaves_out_the_evaluation() {
    let dir = Dir::new();
    let sets = sets(&dir.0);
    let job = job(&dir.0);
    let delay = Duration::from_millis(60);
    for read_ahead in [ReadAhead::Off, ReadAhead::Always] {
        let rt = runtimes();
        let mut bench = Bench::new(&sets, 0, 1, PARAMS);
        bench.read_ahead = read_ahead;
        bench.slow_eval = Some(delay);
        let r = bench.run(&rt, &job, &job.device).unwrap();
        let n = r.per_image.len() as f64;
        let loop_ms = n / r.throughput_fps * 1e3;
        // The served stages of every run, measured inside it.
        let served_ms = r.stages_ms.total.mean * n;
        let eval_ms = delay.as_secs_f64() * 1e3 * n;
        assert!(
            loop_ms < served_ms + eval_ms / 2.0,
            "{read_ahead:?}: loop {loop_ms:.1} ms, stages {served_ms:.1} ms, evaluation {eval_ms:.0} ms"
        );
    }
}

/// With read ahead an iteration lasts at least its frame's decode: the reader's head start
/// during the untimed evaluation is not counted as free decode time.
#[test]
fn an_iteration_lasts_at_least_its_decode() {
    let ms = Duration::from_millis;
    assert_eq!(super::run::iteration_time(ms(2), ms(10), 5.0), ms(12));
    assert_eq!(super::run::iteration_time(ms(0), ms(10), 25.0), ms(25));
    assert_eq!(super::run::iteration_time(ms(3), ms(4), f64::NAN), ms(7));
    assert_eq!(super::run::per_minute(15, Duration::from_secs(30)), 30.0);
    assert_eq!(super::run::per_minute(5, Duration::ZERO), 0.0);
}

/// The warm-up rule: at least `warmup` runs and the minimum time, at most the maximum time or
/// max(`warmup`, the run cap) runs; none for 0.
#[test]
fn the_warmup_rule() {
    let r = WarmupRule::DEFAULT;
    let s = Duration::from_secs;
    assert!(r.done(0, 0, s(0)));
    assert!(
        !r.done(3, 3, Duration::from_millis(500)),
        "the minimum time"
    );
    assert!(!r.done(3, 2, s(5)), "the minimum count");
    assert!(r.done(3, 3, s(2)));
    assert!(r.done(3, 50, s(20)), "the time cap");
    assert!(r.done(3, 200, s(1)), "the run cap");
    assert!(
        !r.done(300, 250, s(5)),
        "a larger warmup raises the run cap"
    );
    assert!(r.done(300, 300, s(1)));
    assert_eq!(
        (
            WARMUP_IMAGES,
            WARMUP_MIN_TIME,
            WARMUP_MAX_TIME,
            WARMUP_MAX_RUNS
        ),
        (5, s(2), s(20), 200)
    );
}

/// The warm-up cycles over the first [`WARMUP_IMAGES`] images (each decoded and served), lasts
/// the minimum time, and is recorded.
#[test]
fn the_warmup_cycles_over_the_first_images() {
    let dir = Dir::new();
    let sets = sets(&dir.0);
    let job = job(&dir.0);
    let seen: Arc<Mutex<Vec<u64>>> = Arc::default();
    let mut script = FakeScript::new(1, INPUT);
    let inner = infer_fn();
    let log = seen.clone();
    script.infer_fn = Some(InferFn(Arc::new(move |chw: &[f32]| {
        let sum: f64 = chw.iter().map(|v| f64::from(*v)).sum();
        log.lock_recover().push(sum.to_bits());
        (inner.0)(chw)
    })));
    let rt = Mutex::new(Runtimes::fake(script));
    let mut bench = Bench::new(&sets, 7, 1, PARAMS);
    bench.warmup_rule = WarmupRule {
        min_time: Duration::from_millis(50),
        ..WarmupRule::DEFAULT
    };
    let r = bench.run(&rt, &job, &job.device).unwrap();
    assert!(r.warmup_runs >= 7, "{}", r.warmup_runs);
    assert!(r.warmup_ms >= 50.0, "{}", r.warmup_ms);
    assert_eq!(r.warmup_iterations, 7);
    let seen = seen.lock_recover().clone();
    let (warm, timed) = seen.split_at(r.warmup_runs);
    assert_eq!(timed.len(), 7);
    // Run k of the warm-up is image k mod 5, the same tensor the timed run of that image saw.
    for (k, v) in warm.iter().enumerate() {
        assert_eq!(*v, timed[k % WARMUP_IMAGES], "warm-up run {k}");
    }
    // No warm-up at all for 0.
    let bench = Bench::new(&sets, 0, 1, PARAMS);
    let r = bench.run(&runtimes(), &job, &job.device).unwrap();
    assert_eq!(r.warmup_runs, 0);
}

/// The live detection requests answered during the timed phase are recorded (the service's
/// metrics); outside the service nothing is; off Windows there is no video-decode reading.
#[test]
fn live_load_during_the_timed_runs_is_recorded() {
    let dir = Dir::new();
    let sets = sets(&dir.0);
    let job = job(&dir.0);
    let served = std::sync::atomic::AtomicU64::new(100);
    let live = || served.fetch_add(15, Ordering::Relaxed);
    let mut bench = Bench::new(&sets, 0, 1, PARAMS);
    bench.live_requests = Some(&live);
    bench.video_decode = true;
    let r = bench.run(&runtimes(), &job, &job.device).unwrap();
    assert_eq!(r.live_requests, Some(15));
    assert!(r.live_requests_per_min.is_some_and(|v| v > 0.0));
    let text = crate::benchmark::report::shared_load_text(&r).unwrap();
    assert!(text.starts_with("measured while serving "), "{text}");
    #[cfg(not(windows))]
    assert_eq!(r.video_decode_pct, None);
    let bench = Bench::new(&sets, 0, 1, PARAMS);
    let r = bench.run(&runtimes(), &job, &job.device).unwrap();
    assert_eq!((r.live_requests, r.live_requests_per_min), (None, None));
    assert_eq!(crate::benchmark::report::shared_load_text(&r), None);
}
