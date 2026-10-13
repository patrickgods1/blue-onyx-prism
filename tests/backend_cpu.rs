//! Real OpenVINO inference on the IPcam-general model: one inference on CPU and on GPU.
//!
//! `#[ignore]`d: needs `models/IPcam-general.{onnx,yaml}` and `openvino/` under
//! `common::test_root()` (a missing one fails the test). Run with
//! `cargo test --test backend_cpu -- --ignored --nocapture` to see timings.

mod common;

use blue_onyx_prism::backend::{CoreOptions, LoadRequest, Runtimes, spec};

#[test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
fn cpu_and_gpu_inference() {
    let root = common::test_root();
    let ov_dir = common::require_openvino();
    let model_path = common::require_ipcam();

    let mut rt = Runtimes::new(&CoreOptions {
        cache_dir: Some(root.join("cache")),
        intra_threads: 0,
        openvino_dir: Some(ov_dir),
    });
    rt.require_any().expect("Runtimes::new");
    let core = rt.openvino().expect("OpenVINO core");
    println!("OpenVINO {}", core.openvino_version());
    println!("available devices: {:?}", core.available_devices());
    for d in core.available_devices().to_vec() {
        println!("  {d}: {}", core.device_full_name(&d));
    }
    let has_gpu = core.has_gpu();
    let req = |requested: &str| LoadRequest {
        path: model_path.clone(),
        requested: requested.into(),
        gpu_precision: None,
    };

    // CPU, no fallback.
    let mut backend = rt
        .load(&spec::parse("CPU").unwrap(), &req("CPU"))
        .expect("load on CPU");
    let info = backend.info().clone();
    println!(
        "CPU: {:?} compile {} ms, inputs {:?}, outputs {:?}",
        info.device, info.compile_ms, info.inputs, info.outputs
    );
    assert_eq!(info.image_input, "images");
    assert_eq!(info.input_size, (640, 640));
    assert_eq!(info.inputs.len(), 1);
    assert_eq!(info.inputs[0].shape, vec![1, 3, 640, 640]);
    assert_eq!(info.outputs.len(), 1);
    assert_eq!(info.outputs[0].shape, vec![1, 25200, 8]);
    assert_eq!(info.device.actual, "CPU");
    assert_eq!(info.device.spec, "openvino:cpu");
    assert!(!info.device.fell_back);

    let chw = vec![0f32; 3 * 640 * 640];
    let t = std::time::Instant::now();
    let outs = backend.infer(&chw, &[]).expect("infer CPU");
    println!("CPU first inference {} ms", t.elapsed().as_millis());
    let t = std::time::Instant::now();
    backend.infer(&chw, &[]).expect("infer CPU");
    println!("CPU second inference {} ms", t.elapsed().as_millis());
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].shape, vec![1, 25200, 8]);
    assert_eq!(outs[0].as_f32().expect("f32 output").len(), 25200 * 8);
    assert!(backend.infer(&chw[1..], &[]).is_err(), "length check");
    // `infer_into` refills the same output buffers.
    let mut reused = Vec::new();
    backend
        .infer_into(&chw, &[], &mut reused)
        .expect("infer_into");
    let ptr = reused[0].as_f32().expect("f32 output").as_ptr();
    backend
        .infer_into(&chw, &[], &mut reused)
        .expect("infer_into");
    assert_eq!(reused[0].as_f32().unwrap().as_ptr(), ptr, "buffer reused");
    assert_eq!(reused, outs, "same input, same outputs");

    // GPU with CPU fallback, loaded twice (second load should hit the compile cache).
    for run in 1..=2 {
        let mut backend = rt
            .load(&spec::parse("GPU").unwrap(), &req("GPU"))
            .expect("load on GPU (with fallback)");
        let dev = &backend.info().device;
        println!(
            "GPU run {run}: {:?} compile {} ms, provider '{}'",
            dev,
            backend.info().compile_ms,
            dev.execution_provider()
        );
        assert_eq!(dev.requested, "GPU");
        if !has_gpu {
            // No GPU listed: CPU, reported as a fallback (unchanged from before 6.1).
            assert_eq!(dev.actual, "CPU");
            assert!(dev.fell_back);
            assert!(dev.execution_provider().ends_with(", fallback)"));
        }
        let t = std::time::Instant::now();
        let outs = backend.infer(&chw, &[]).expect("infer GPU");
        let first = t.elapsed().as_millis();
        let t = std::time::Instant::now();
        backend.infer(&chw, &[]).expect("infer GPU");
        println!(
            "GPU run {run}: first inference {first} ms, second {} ms",
            t.elapsed().as_millis()
        );
        assert_eq!(outs[0].as_f32().expect("f32 output").len(), 25200 * 8);
    }

    // `Runtimes::new` configures no ONNX Runtime library here, so the `ort:cuda` spec falls back to
    // OpenVINO CPU (the ORT backend itself is covered by tests/backend_ort.rs).
    let backend = rt
        .load(&spec::parse("ort:cuda").unwrap(), &req("ort:cuda"))
        .expect("ort:cuda falls back to CPU");
    let dev = &backend.info().device;
    assert_eq!((dev.actual.as_str(), dev.fell_back), ("CPU", true));
    assert_eq!(dev.requested, "ort:cuda");
}
