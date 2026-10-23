//! A config file edited on disk while the service runs applies by itself, with no page open:
//! the per-generation disk watch (`config_store::spawn_disk_watch`, every
//! `config_store::DISK_POLL`) picks it up and applies it like a Save (live settings swap,
//! per-model reload). Before it, only `GET /v1/config` and the Config / Benchmark pages looked
//! at the file, so an edit made while only Blue Iris was calling never applied.
//!
//! The test polls only routes that never read the config file (`/v1/vision/custom/list`,
//! detections), so nothing but the watch can apply the edit.
//!
//! `#[ignore]`d: needs `models/IPcam-general.{onnx,yaml}` and `openvino/` under
//! `common::test_root()`. Run with `cargo test --test config_disk_http -- --ignored --nocapture`.

mod common;

use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::model::ModelFamilyKind;
use common::{Server, list_models, post_image, temp_dir, wait_until};
use std::time::{Duration, Instant};

fn ipcam(name: &str, path: &std::path::Path) -> ModelConfig {
    ModelConfig {
        name: Some(name.into()),
        path: path.to_path_buf(),
        family: ModelFamilyKind::Yolo5,
        ..Default::default()
    }
}

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn edit_on_disk_applies_without_a_page() {
    let model = common::require_ipcam();
    let ov_dir = common::require_openvino();
    let image = common::test_image();
    let config = Config {
        cache_dir: common::cache_dir("disk-edit"),
        openvino_dir: Some(ov_dir),
        device: "CPU".into(),
        log_level: LogLevel::Warn,
        auto_download: false,
        confidence_threshold: 0.9,
        models: vec![ipcam("ipcam-a", &model), ipcam("ipcam-b", &model)],
        ..Default::default()
    };
    let dir = temp_dir("disk-edit");
    let path = dir.join("cfg.json");
    let served = Server::start(config, path.clone()).await;
    let base = served.base.clone();
    let client = reqwest::Client::new();
    let both = vec!["ipcam-a".to_string(), "ipcam-b".to_string()];
    wait_until("both models listed", Duration::from_secs(60), || async {
        list_models(&client, &base).await.as_ref() == Some(&both)
    })
    .await;
    let detect = format!("{base}/v1/vision/custom/ipcam-a");
    wait_until("ipcam-a to answer", Duration::from_secs(300), || async {
        post_image(&client, &detect, &image).await.success
    })
    .await;
    let high = post_image(&client, &detect, &image).await;

    // Edit in place (truncate + write, like most editors): ipcam-b off, threshold down.
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    v["models"][1]["enabled"] = false.into();
    v["confidence_threshold"] = 0.1.into();
    let t = Instant::now();
    std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();

    let only_a = vec!["ipcam-a".to_string()];
    wait_until(
        "ipcam-b to stop after the edit on disk",
        Duration::from_secs(15),
        || async { list_models(&client, &base).await.as_ref() == Some(&only_a) },
    )
    .await;
    let disabled_in = t.elapsed();
    let low = post_image(&client, &detect, &image).await;
    eprintln!(
        "edit on disk applied in {disabled_in:?}; ipcam-a: {} detections at 0.9, {} at 0.1",
        high.count, low.count
    );
    assert!(low.success && low.count > high.count, "{high:?} / {low:?}");

    // A broken edit is reported and changes nothing; fixing the file applies it.
    std::fs::write(&path, "{ \"models\": [").unwrap();
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(list_models(&client, &base).await, Some(only_a.clone()));
    let cfg = common::get_json(&client, &format!("{base}/v1/config"))
        .await
        .expect("GET /v1/config");
    eprintln!("fileError: {}", cfg["fileError"]);
    assert!(
        cfg["fileError"]
            .as_str()
            .unwrap_or_default()
            .contains("not applied"),
        "{cfg}"
    );
    v["models"][1]["enabled"] = true.into();
    std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    wait_until(
        "ipcam-b to start again after the file is fixed",
        Duration::from_secs(15),
        || async { list_models(&client, &base).await.as_ref() == Some(&both) },
    )
    .await;
    served.stop().await;
}
