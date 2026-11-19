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
    bench.read_ahead = read_ahead;
    bench.on_image = Some(&sink);
    bench.cancel = Some(&cancel);
    bench.max_stored_preds = 300;
    let r = bench.run(&rt, job, &job.device);
    drop(bench);
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
