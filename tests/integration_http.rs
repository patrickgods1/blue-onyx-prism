//! End-to-end HTTP test: real OpenVINO, real model, real multipart upload.
//!
//! `#[ignore]`d: needs `models/IPcam-general.{onnx,yaml}` and `openvino/` under
//! `common::test_root()` (`BLUE_ONYX_TEST_ROOT`, else the repo root); a missing fixture fails
//! the test. Run with `cargo test -- --ignored`.

mod common;

use blue_onyx_prism::config::{Config, ModelConfig};
use blue_onyx_prism::model::ModelFamilyKind;
use common::{Server, post_image_with};
use std::time::{Duration, Instant};

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn detection_over_http() {
    let model = common::require_ipcam();
    let ov_dir = common::require_openvino();
    let image = common::test_image();

    let cache = common::cache_dir("it");
    let config = Config {
        force_cpu: true,
        cache_dir: cache,
        openvino_dir: Some(ov_dir),
        models: vec![ModelConfig {
            path: model,
            family: ModelFamilyKind::Yolo5,
            ..Default::default()
        }],
        ..Default::default()
    };
    let server = Server::start_in_process(config).await;
    let base = server.base.clone();
    let client = reqwest::Client::new();
    let post = |url: String| {
        let (client, image) = (client.clone(), image.clone());
        async move { post_image_with(&client, &url, &image, Some("0.4")).await }
    };

    // Wait for the model to compile (GET / shows the state).
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if let Ok(r) = client.get(format!("{base}/")).send().await
            && let Ok(body) = r.text().await
        {
            if body.contains("<td>Ready</td>") {
                break;
            }
            assert!(!body.contains("Failed:"), "model failed to load:\n{body}");
        }
        assert!(
            Instant::now() < deadline,
            "model did not become ready in time"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    let r = post(format!("{base}/v1/vision/detection")).await;
    eprintln!("default: {r:?}");
    assert!(r.success, "detection failed: {r:?}");
    assert!(!r.predictions.is_empty(), "no predictions: {r:?}");
    assert_eq!(r.count as usize, r.predictions.len());
    assert_eq!(r.command, "detect");

    let r = post(format!("{base}/v1/vision/custom/IPcam-general")).await;
    assert!(r.success, "custom detection failed: {r:?}");
    assert_eq!(r.command, "custom");

    let r = post(format!("{base}/v1/vision/custom/nope")).await;
    assert!(!r.success);
    assert!(r.error.unwrap_or_default().contains("Unknown model"));

    let list = common::list_models(&client, &base)
        .await
        .expect("model list");
    assert!(list.iter().any(|m| m == "IPcam-general"));

    server.stop().await;
}
