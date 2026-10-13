//! Config changes applied without a restart (docs/REVIEW_PLAN.md, phase 3), through the
//! production run loop (`runner::run_server_on`) and real OpenVINO:
//! - a global threshold saved with `POST /config` is used by the next detection;
//! - disabling a model on the Models card drops it from `/v1/vision/custom/list` at once and
//!   enabling it again loads and serves it;
//! - a per-model device change reloads only that model (`/stats.json` shows the reload, the
//!   generation in `/v1/config` stays the same).
//!
//! `#[ignore]`d: needs `models/IPcam-general.{onnx,yaml}` and `openvino/` under
//! `common::test_root()` (a missing one fails the test).

mod common;

use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::model::ModelFamilyKind;
use common::{
    Server, TEST_IMAGE, get_json, list_models, post_form_json as post_form, post_image, temp_dir,
    wait_until,
};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The model, OpenVINO and the test image; panics when a fixture is missing.
fn fixtures() -> (PathBuf, PathBuf, Vec<u8>) {
    (
        common::require_ipcam(),
        common::require_openvino(),
        common::test_image(),
    )
}

fn config_with(ov_dir: &Path, models: Vec<ModelConfig>) -> Config {
    Config {
        cache_dir: common::cache_dir("live"),
        openvino_dir: Some(ov_dir.to_path_buf()),
        // OpenVINO CPU in the legacy spelling (no GPU, nothing to download on any platform);
        // the device test moves one model to the explicit `openvino:cpu`.
        device: "CPU".into(),
        log_level: LogLevel::Warn,
        // Nothing to download in these tests.
        auto_download: false,
        models,
        ..Default::default()
    }
}

fn ipcam(name: &str, path: &Path) -> ModelConfig {
    ModelConfig {
        name: Some(name.into()),
        path: path.to_path_buf(),
        family: ModelFamilyKind::Yolo5,
        ..Default::default()
    }
}

/// The `/stats.json` entry of `model`.
async fn stats_of(client: &reqwest::Client, base: &str, model: &str) -> Option<serde_json::Value> {
    let v = get_json(client, &format!("{base}/stats.json")).await?;
    v["models"]
        .as_array()?
        .iter()
        .find(|m| m["name"] == model)
        .cloned()
}

/// Wait until `/v1/vision/custom/list` is exactly `names` and every one of them is Ready.
async fn wait_ready(client: &reqwest::Client, base: &str, names: &[&str]) {
    let expected: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    wait_until(
        &format!("{names:?} to be served"),
        Duration::from_secs(60),
        || async { list_models(client, base).await.as_ref() == Some(&expected) },
    )
    .await;
    for name in names {
        wait_until(
            &format!("{name} to be ready"),
            Duration::from_secs(300),
            || async {
                let s = stats_of(client, base, name).await.unwrap_or_default();
                let st = s["state"].as_str().unwrap_or_default();
                assert!(!st.starts_with("Failed"), "{name} failed: {s}");
                s["state"] == "Ready"
            },
        )
        .await;
    }
}

async fn generation(client: &reqwest::Client, base: &str) -> u64 {
    get_json(client, &format!("{base}/v1/config"))
        .await
        .expect("GET /v1/config")["generation"]
        .as_u64()
        .expect("generation")
}

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn threshold_applies_to_the_next_detection() {
    let (model, ov_dir, image) = fixtures();
    let config = Config {
        confidence_threshold: 0.9,
        ..config_with(&ov_dir, vec![ipcam("IPcam-general", &model)])
    };
    let client = reqwest::Client::new();
    let served = Server::start(config, temp_dir("thr").join("cfg.json")).await;
    let base = served.base.clone();
    wait_ready(&client, &base, &["IPcam-general"]).await;
    let gen0 = generation(&client, &base).await;
    let detect = format!("{base}/v1/vision/detection");

    let high = post_image(&client, &detect, &image).await;
    assert!(high.success, "{high:?}");
    assert!(
        high.predictions.iter().all(|p| p.confidence >= 0.9),
        "{high:?}"
    );

    let t = Instant::now();
    let j = post_form(
        &client,
        &format!("{base}/config"),
        "confidence_threshold=0.1&request_timeout_secs=15",
    )
    .await;
    let msg = j["message"].as_str().unwrap_or_default();
    assert!(msg.contains("Applied now"), "{j}");
    assert!(msg.contains("Confidence threshold"), "{j}");
    assert_eq!(j["restarting"], false);
    let low = post_image(&client, &detect, &image).await;
    let applied_in = t.elapsed();
    assert!(low.success, "{low:?}");
    eprintln!(
        "IPcam-general on {TEST_IMAGE}: {} detections at 0.9, {} at 0.1 (applied and detected in {applied_in:?})",
        high.count, low.count
    );
    assert!(
        low.count > high.count,
        "0.1 must find more than 0.9: {} vs {}",
        low.count,
        high.count
    );
    assert!(
        low.predictions.iter().any(|p| p.confidence < 0.9),
        "{low:?}"
    );

    // No restart: same generation, same worker (no reload either), nothing pending.
    assert_eq!(generation(&client, &base).await, gen0);
    let s = stats_of(&client, &base, "IPcam-general").await.unwrap();
    assert_eq!(s["reloads"], 0, "{s}");
    let cfg = get_json(&client, &format!("{base}/v1/config"))
        .await
        .unwrap();
    assert_eq!(cfg["pendingCount"], 0, "{cfg}");
    assert_eq!(cfg["models"][0]["running"]["threshold"], 0.1, "{cfg}");
    served.stop().await;
}

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn disable_and_enable_without_restart() {
    let (model, ov_dir, image) = fixtures();
    let config = config_with(
        &ov_dir,
        vec![ipcam("ipcam-a", &model), ipcam("ipcam-b", &model)],
    );
    let client = reqwest::Client::new();
    let served = Server::start(config, temp_dir("enable").join("cfg.json")).await;
    let base = served.base.clone();
    wait_ready(&client, &base, &["ipcam-a", "ipcam-b"]).await;
    let gen0 = generation(&client, &base).await;

    // Disable ipcam-b: gone from the list at once (10 s allowed so a loaded machine cannot fail it).
    let t = Instant::now();
    let j = post_form(
        &client,
        &format!("{base}/config/models"),
        "enabled=ipcam-a&default_model=ipcam-a&action=save",
    )
    .await;
    assert!(
        j["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Stopping ipcam-b"),
        "{j}"
    );
    wait_until(
        "ipcam-b to leave the list",
        Duration::from_secs(10),
        || async {
            list_models(&client, &base).await.as_deref() == Some(&["ipcam-a".to_string()][..])
        },
    )
    .await;
    eprintln!("ipcam-b dropped from the list after {:?}", t.elapsed());
    let r = post_image(&client, &format!("{base}/v1/vision/custom/ipcam-b"), &image).await;
    assert!(!r.success);
    assert!(
        r.error
            .as_deref()
            .unwrap_or_default()
            .contains("Unknown model"),
        "{r:?}"
    );
    // ipcam-a kept serving.
    let r = post_image(&client, &format!("{base}/v1/vision/custom/ipcam-a"), &image).await;
    assert!(r.success && !r.predictions.is_empty(), "{r:?}");

    // Enable it again: it loads and serves.
    let t = Instant::now();
    let j = post_form(
        &client,
        &format!("{base}/config/models"),
        "enabled=ipcam-a&enabled=ipcam-b&default_model=ipcam-a&action=save",
    )
    .await;
    assert!(
        j["message"]
            .as_str()
            .unwrap_or_default()
            .contains("Loading ipcam-b on"),
        "{j}"
    );
    wait_ready(&client, &base, &["ipcam-a", "ipcam-b"]).await;
    eprintln!("ipcam-b ready again after {:?}", t.elapsed());
    let r = post_image(&client, &format!("{base}/v1/vision/custom/ipcam-b"), &image).await;
    assert!(r.success && !r.predictions.is_empty(), "{r:?}");
    assert_eq!(generation(&client, &base).await, gen0, "no restart");
    served.stop().await;
}

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn device_change_reloads_only_that_model() {
    let (model, ov_dir, image) = fixtures();
    let config = config_with(
        &ov_dir,
        vec![ipcam("ipcam-a", &model), ipcam("ipcam-b", &model)],
    );
    let client = reqwest::Client::new();
    let served = Server::start(config, temp_dir("device").join("cfg.json")).await;
    let base = served.base.clone();
    wait_ready(&client, &base, &["ipcam-a", "ipcam-b"]).await;
    let gen0 = generation(&client, &base).await;
    let before = stats_of(&client, &base, "ipcam-a").await.unwrap();
    assert_eq!(before["requestedDevice"], "CPU", "{before}");
    assert_eq!(before["reloads"], 0);

    // The Models card's device select for ipcam-a only.
    let t = Instant::now();
    let j = post_form(
        &client,
        &format!("{base}/config/models"),
        "enabled=ipcam-a&enabled=ipcam-b&default_model=ipcam-a&device.ipcam-a=openvino:cpu\
         &device.ipcam-b=&action=save",
    )
    .await;
    let msg = j["message"].as_str().unwrap_or_default();
    assert!(msg.contains("Reloading ipcam-a on"), "{j}");
    assert!(!msg.contains("ipcam-b"), "only ipcam-a reloads: {j}");
    assert_eq!(j["restarting"], false);
    wait_until("ipcam-a to reload", Duration::from_secs(120), || async {
        let s = stats_of(&client, &base, "ipcam-a")
            .await
            .unwrap_or_default();
        s["reloads"] == 1 && s["state"] == "Ready"
    })
    .await;
    eprintln!("ipcam-a reloaded on openvino:cpu in {:?}", t.elapsed());
    let a = stats_of(&client, &base, "ipcam-a").await.unwrap();
    assert_eq!(a["requestedDevice"], "openvino:cpu", "{a}");
    assert_eq!(a["device"], "CPU", "{a}");
    let b = stats_of(&client, &base, "ipcam-b").await.unwrap();
    assert_eq!(b["reloads"], 0, "{b}");
    assert_eq!(b["requestedDevice"], "CPU", "{b}");
    // Same generation, nothing pending, the change is logged.
    let cfg = get_json(&client, &format!("{base}/v1/config"))
        .await
        .unwrap();
    assert_eq!(cfg["generation"].as_u64(), Some(gen0), "{cfg}");
    assert_eq!(cfg["pendingCount"], 0, "{cfg}");
    let last = cfg["changes"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["source"], "config page (Models card)", "{last}");
    let r = post_image(&client, &format!("{base}/v1/vision/custom/ipcam-a"), &image).await;
    assert!(r.success && !r.predictions.is_empty(), "{r:?}");
    served.stop().await;
}

/// Reloading one model over and over must not accumulate threads or memory.
#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn reload_model_n_times_does_not_leak() {
    let (model, ov_dir, image) = fixtures();
    let config = config_with(&ov_dir, vec![ipcam("ipcam-a", &model)]);
    let client = reqwest::Client::new();
    let served = Server::start(config, temp_dir("leak").join("cfg.json")).await;
    let base = served.base.clone();
    wait_ready(&client, &base, &["ipcam-a"]).await;

    // Alternate the device spelling: each change reloads only this model.
    let reload = |n: u64| {
        let client = client.clone();
        let base = base.clone();
        async move {
            let dev = if n % 2 == 1 { "openvino:cpu" } else { "CPU" };
            let j = post_form(
                &client,
                &format!("{base}/config/models"),
                &format!("enabled=ipcam-a&default_model=ipcam-a&device.ipcam-a={dev}&action=save"),
            )
            .await;
            let msg = j["message"].as_str().unwrap_or_default();
            assert!(msg.contains("Reloading ipcam-a on"), "{j}");
            wait_until("ipcam-a to reload", Duration::from_secs(120), || async {
                let s = stats_of(&client, &base, "ipcam-a")
                    .await
                    .unwrap_or_default();
                s["reloads"] == n && s["state"] == "Ready"
            })
            .await;
        }
    };

    reload(1).await;
    let threads0 = common::thread_count();
    let rss0 = common::rss_kb();
    for n in 2..=10 {
        reload(n).await;
    }
    let threads1 = common::thread_count();
    let rss1 = common::rss_kb();
    eprintln!("after reload 1: threads {threads0:?}, rss {rss0:?} KiB");
    eprintln!("after reload 10: threads {threads1:?}, rss {rss1:?} KiB");
    match (threads0, threads1) {
        (Some(a), Some(b)) => assert!(b <= a + 2, "threads grew from {a} to {b}"),
        _ => eprintln!("thread count unavailable on this platform: assertion skipped"),
    }
    if let (Some(a), Some(b)) = (rss0, rss1) {
        assert!(b * 4 <= a * 5, "RSS grew more than 25%: {a} -> {b} KiB");
    }

    let s = stats_of(&client, &base, "ipcam-a").await.unwrap();
    assert_eq!(s["reloads"], 10, "{s}");
    let r = post_image(&client, &format!("{base}/v1/vision/custom/ipcam-a"), &image).await;
    assert!(r.success && !r.predictions.is_empty(), "{r:?}");
    served.stop().await;
}
