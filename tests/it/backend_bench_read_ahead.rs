//! The benchmark's read ahead on real OpenVINO CPU inference (IPcam-general): the pipelined
//! timed loop gives bit-identical predictions to the serial loop, and the wall-clock time per
//! image of both is printed.
//!
//! `#[ignore]`d: needs `models/IPcam-general.{onnx,yaml}` and `openvino/` under
//! `common::test_root()`. CPU only (`openvino:cpu`), never a GPU. Images: the `.jpg` / `.png`
//! files of `$BOP_READ_AHEAD_IMAGES` (sorted, at most `$BOP_READ_AHEAD_MAX`, default 40), else
//! 12 synthetic 2688x1520 JPEGs (a camera's size). Run with
//! `cargo test --test it backend_bench_read_ahead -- --ignored --nocapture --test-threads=1`.

use crate::common;

use blue_onyx_prism::backend::{CoreOptions, Runtimes};
use blue_onyx_prism::benchmark::images::{GroundTruth, ImageSet, SetImage, SetKind};
use blue_onyx_prism::benchmark::{Bench, Job, ReadAhead, RunResult};
use blue_onyx_prism::model::{ModelFamilyKind, PostParams};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

type Bits = Vec<Vec<(String, u32, [u32; 4])>>;

/// Stored and evaluation predictions of every image, as exact bits, in order.
fn bits(r: &RunResult) -> (Vec<String>, Bits, Bits) {
    let files = r.per_image.iter().map(|i| i.file.clone()).collect();
    let stored = r
        .per_image
        .iter()
        .map(|i| {
            i.preds
                .iter()
                .map(|p| {
                    (
                        p.label.clone(),
                        p.confidence.to_bits(),
                        p.bbox.map(f32::to_bits),
                    )
                })
                .collect()
        })
        .collect();
    let eval = r
        .eval_preds
        .iter()
        .map(|v| {
            v.iter()
                .map(|p| {
                    (
                        p.label.clone(),
                        p.confidence.to_bits(),
                        p.bbox.map(f32::to_bits),
                    )
                })
                .collect()
        })
        .collect();
    (files, stored, eval)
}

fn set_image(path: PathBuf) -> SetImage {
    let (width, height) = image::image_dimensions(&path).unwrap_or((0, 0));
    SetImage {
        file: path.file_name().unwrap().to_string_lossy().into_owned(),
        path: Some(path),
        width,
        height,
        tags: Vec::new(),
        gt: None,
    }
}

/// The images of `$BOP_READ_AHEAD_IMAGES`, else synthetic camera-sized JPEGs in `scratch`.
fn images(scratch: &Path) -> Vec<SetImage> {
    if let Some(dir) = std::env::var_os("BOP_READ_AHEAD_IMAGES") {
        let max: usize = std::env::var("BOP_READ_AHEAD_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40);
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("BOP_READ_AHEAD_IMAGES")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                    matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png")
                })
            })
            .collect();
        files.sort();
        files.truncate(max);
        return files.into_iter().map(set_image).collect();
    }
    (0..12u32)
        .map(|k| {
            let img = image::RgbImage::from_fn(2688, 1520, |x, y| {
                let v = (x / 8 + y / 8 + k * 37) ^ x.wrapping_mul(y + k);
                image::Rgb([
                    (v % 256) as u8,
                    ((x + k * 29) % 256) as u8,
                    ((y * 3) % 256) as u8,
                ])
            });
            let path = scratch.join(format!("frame-{k:02}.jpg"));
            img.save(&path).unwrap();
            set_image(path)
        })
        .collect()
}

#[test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
fn backend_bench_read_ahead_matches_serial_on_cpu() {
    let root = common::test_root();
    let ov_dir = common::require_openvino();
    let model = common::require_ipcam();
    let scratch = common::temp_dir("read-ahead");
    let images = images(&scratch);
    assert!(!images.is_empty(), "no images");
    let sets = [ImageSet {
        id: "read-ahead".into(),
        title: "read ahead".into(),
        kind: SetKind::Dir,
        dir: None,
        license: String::new(),
        attribution: String::new(),
        source: String::new(),
        ground_truth: GroundTruth::None,
        scored_labels: None,
        transform: None,
        images,
    }];
    let classes =
        blue_onyx_prism::model::classes::load_class_names(&model.with_extension("yaml")).unwrap();
    let job = Job {
        name: "IPcam-general".into(),
        path: model.clone(),
        family: ModelFamilyKind::Yolo5,
        classes,
        device: "openvino:cpu".into(),
        gpu_precision: None,
        confidence_threshold: None,
        object_filter: None,
        variant: None,
    };
    let rt = Runtimes::new(&CoreOptions {
        cache_dir: Some(root.join("cache")),
        intra_threads: 0,
        openvino_dir: Some(ov_dir),
    });
    rt.require_any().expect("runtimes");
    let rt = Mutex::new(rt);
    let params = PostParams {
        confidence_threshold: 0.4,
        nms_iou: 0.45,
    };
    let run = |read_ahead: ReadAhead| {
        let mut bench = Bench::new(&sets, 2, 1, params);
        bench.read_ahead = read_ahead;
        bench.max_stored_preds = usize::MAX;
        bench.run(&rt, &job, "openvino:cpu").expect("benchmark run")
    };
    // Serial, read ahead, serial, read ahead: order effects (warm caches) show up as a spread.
    let mut serial = Vec::new();
    let mut ahead = Vec::new();
    for _ in 0..2 {
        serial.push(run(ReadAhead::Off));
        ahead.push(run(ReadAhead::Always));
    }
    let reference = bits(&serial[0]);
    let boxes: usize = reference.1.iter().map(Vec::len).sum();
    println!(
        "{} images, {boxes} stored boxes, {} evaluation boxes",
        reference.0.len(),
        reference.2.iter().map(Vec::len).sum::<usize>()
    );
    for r in serial.iter().chain(&ahead) {
        assert_eq!(bits(r), reference, "bit-identical predictions");
        assert_eq!(r.device, "CPU");
    }
    for (name, runs) in [("serial", &serial), ("read ahead", &ahead)] {
        for r in runs.iter() {
            let s = &r.stages_ms;
            println!(
                "{name:<10} pipelined={:<5} wall {:7.1} ms/image ({:5.2} img/s) | p50 decode {:6.1} \
                 preprocess {:6.1} infer {:6.1} postprocess {:5.1} total {:6.1} ms",
                r.pipelined,
                1000.0 / r.throughput_fps,
                r.throughput_fps,
                s.decode.p50,
                s.preprocess.p50,
                s.infer.p50,
                s.postprocess.p50,
                s.total.p50
            );
        }
    }
    assert!(serial.iter().all(|r| !r.pipelined) && ahead.iter().all(|r| r.pipelined));
}
