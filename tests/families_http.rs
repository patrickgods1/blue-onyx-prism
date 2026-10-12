//! End-to-end HTTP test for the model families: YOLO26 (IR from `scripts/export_model.py`,
//! both explicit and `auto` family) and RT-DETRv2 (ONNX), served together on CPU and queried
//! through `/v1/vision/custom/{model}`.
//!
//! `#[ignore]`d: needs `openvino/` and `models/IPcam-general.{onnx,yaml}` under
//! `common::test_root()` (a missing one fails the test). `yolo26s.xml`, `yolo26n.xml` and
//! `rt-detrv2-s.onnx` are optional extras: each present one is served and checked too.

mod common;

use blue_onyx_prism::config::{Config, ModelConfig};
use blue_onyx_prism::model::ModelFamilyKind;
use common::{Server, post_image_with};
use std::time::{Duration, Instant};

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn yolo26_and_rtdetr_over_http() {
    let ov_dir = common::require_openvino();
    let ipcam = common::require_ipcam();
    let candidates = [
        ("yolo26s", "yolo26s.xml", ModelFamilyKind::Yolo26),
        ("yolo26n-auto", "yolo26n.xml", ModelFamilyKind::Auto),
        ("rt-detrv2-s", "rt-detrv2-s.onnx", ModelFamilyKind::RtDetr),
    ];
    // The required model first (IPcam classes, not COCO), then the optional COCO ones.
    let mut models = vec![ModelConfig {
        name: Some("IPcam-general".into()),
        path: ipcam,
        family: ModelFamilyKind::Yolo5,
        ..Default::default()
    }];
    models.extend(candidates.iter().filter_map(|(name, file, family)| {
        common::optional_model(file).map(|path| ModelConfig {
            name: Some(name.to_string()),
            path,
            family: *family,
            ..Default::default()
        })
    }));
    let names: Vec<String> = models.iter().map(|m| m.name.clone().unwrap()).collect();
    let image = common::test_image();

    let config = Config {
        force_cpu: true,
        cache_dir: common::cache_dir("fam"),
        openvino_dir: Some(ov_dir),
        default_model: names.first().cloned(),
        models,
        ..Default::default()
    };
    let server = Server::start_in_process(config).await;
    let base = server.base.clone();
    let client = reqwest::Client::new();

    // Wait until every model is compiled (GET / shows one state cell per model).
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if let Ok(r) = client.get(format!("{base}/")).send().await
            && let Ok(body) = r.text().await
        {
            assert!(!body.contains("Failed:"), "a model failed to load:\n{body}");
            if body.matches("<td>Ready</td>").count() == names.len() {
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "models did not become ready in time"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    for name in &names {
        let url = format!("{base}/v1/vision/custom/{name}");
        let r = post_image_with(&client, &url, &image, Some("0.4")).await;
        eprintln!("{name}: {r:?}");
        assert!(r.success, "{name}: detection failed: {r:?}");
        if name == "IPcam-general" {
            assert!(!r.predictions.is_empty(), "{name}: nothing detected: {r:?}");
        } else {
            for want in ["dog", "bicycle"] {
                assert!(
                    r.predictions.iter().any(|p| p.label == want),
                    "{name}: no '{want}' in {:?}",
                    r.predictions
                );
            }
        }
        for p in &r.predictions {
            assert!(
                p.x_min < p.x_max && p.y_min < p.y_max,
                "{name}: bad box {p:?}"
            );
            assert!(p.confidence >= 0.4, "{name}: below min_confidence {p:?}");
        }
    }

    server.stop().await;
}
