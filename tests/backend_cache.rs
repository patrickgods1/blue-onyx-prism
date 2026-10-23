//! The OpenVINO model cache keeps one folder per device and inference precision, so a model that
//! moves to f32 (`catalog::F32_MODELS`) never imports a blob compiled in f16.
//!
//! `#[ignore]`d: needs `models/IPcam-general.onnx` and `openvino/` (see tests/common). Run with
//! `cargo test --test backend_cache -- --ignored --nocapture`.

mod common;

use blue_onyx_prism::backend::{CoreOptions, LoadRequest, Runtimes, spec};
use std::path::Path;

/// `(folder name, blob count)` for every folder under `root` holding OpenVINO blobs, plus
/// `("", n)` for blobs in the root itself.
fn blobs(root: &Path) -> Vec<(String, usize)> {
    let count = |d: &Path| {
        std::fs::read_dir(d).map_or(0, |r| {
            r.flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(".blob"))
                .count()
        })
    };
    let mut out = vec![(String::new(), count(root))];
    for e in std::fs::read_dir(root).unwrap().flatten() {
        if e.path().is_dir() {
            out.push((
                e.file_name().to_string_lossy().into_owned(),
                count(&e.path()),
            ));
        }
    }
    out.retain(|(_, n)| *n > 0);
    out.sort();
    out
}

#[test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
fn precision_change_never_reuses_a_blob() {
    let ov_dir = common::require_openvino();
    let ipcam = common::require_ipcam();
    let dir = common::temp_dir("ov-cache-precision");
    let cache = dir.join("cache");
    // The same graph under an F32_MODELS file name: OpenVINO hashes the graph, not the path, so
    // without per-precision folders this copy imported the f16 blob of the original.
    let f32_copy = dir.join("exdark-rfdetr-nano.onnx");
    std::fs::copy(&ipcam, &f32_copy).unwrap();

    let mut rt = Runtimes::new(&CoreOptions {
        cache_dir: Some(cache.clone()),
        intra_threads: 0,
        openvino_dir: Some(ov_dir),
    });
    rt.require_any().expect("Runtimes::new");
    let cpu = spec::parse("openvino:cpu").unwrap();
    let req = |path: &Path| LoadRequest {
        path: path.to_path_buf(),
        requested: "openvino:cpu".into(),
        gpu_precision: None,
    };
    // A non-trivial input so f16 and f32 differ in the outputs.
    let chw: Vec<f32> = (0..3 * 640 * 640u64)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    let mut run = |path: &Path| {
        let mut b = rt.load(&cpu, &req(path)).expect("load on CPU");
        let ms = b.info().compile_ms;
        let out = b.infer(&chw, &[]).expect("infer")[0]
            .as_f32()
            .expect("f32 output")
            .to_vec();
        (out, ms)
    };

    let (default_out, ms) = run(&ipcam);
    println!(
        "IPcam-general (CPU default precision): compile {ms} ms; blobs {:?}",
        blobs(&cache)
    );
    let (f32_out, ms) = run(&f32_copy);
    println!(
        "same graph as an F32 model: compile {ms} ms; blobs {:?}",
        blobs(&cache)
    );
    let (again, ms) = run(&ipcam);
    println!("IPcam-general again (cache hit): compile {ms} ms");
    let max_diff = default_out
        .iter()
        .zip(&f32_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "max |default - f32| over {} outputs: {max_diff}",
        f32_out.len()
    );

    let found = blobs(&cache);
    assert!(
        found.iter().all(|(d, _)| d.starts_with("openvino-cpu-")),
        "blobs outside the per-precision folders: {found:?}"
    );
    assert!(
        found.iter().any(|(d, _)| d == "openvino-cpu-f32"),
        "no f32 cache folder: {found:?}"
    );
    assert_eq!(
        again, default_out,
        "the cached default-precision blob is reused"
    );
    if found.len() == 2 {
        // The CPU default is not f32 (f16 on ARM, bf16 on some x86): two distinct entries, and
        // the f32 model really ran in f32.
        assert!(
            max_diff > 0.0,
            "f32 run produced the f16 outputs: stale blob"
        );
    } else {
        // The CPU default is f32 already (most x86): one folder, the same blob is correct.
        assert_eq!(found.len(), 1, "{found:?}");
    }
}
