//! `enabled: false` models stay in the config but are not loaded or served; with every model
//! disabled the HTTP server (and its web UI) still comes up.
//!
//! Switching models through the Models card (`POST /config/models`, with a restart and with a
//! per-model reload) and the in-process restart (`runner::run_server_on`, as used by the binary
//! and the Windows service) is covered too.
//!
//! Real OpenVINO, `#[ignore]`d: needs `models/IPcam-general.{onnx,yaml}` and `openvino/` under
//! `common::test_root()` (a missing one fails the test). The switch test serves
//! `models/rt-detrv2-s.onnx` as its second model when present; without it the entry named
//! `rt-detrv2-s` is backed by a second copy of IPcam-general (the switching logic is the same,
//! only the COCO label check is skipped).

mod common;

use blue_onyx_prism::api::VisionDetectionResponse;
use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::model::ModelFamilyKind;
use common::{Server, list_models, post_form, post_image, temp_dir, wait_until};
use std::path::Path;
use std::time::{Duration, Instant};

/// GET `path`, retrying until the listener is up.
async fn get_text(client: &reqwest::Client, url: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(r) = client.get(url).send().await
            && let Ok(body) = r.text().await
        {
            return body;
        }
        assert!(Instant::now() < deadline, "server did not come up");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn disabled_models_are_not_served() {
    let model = common::require_ipcam();
    let ov_dir = common::require_openvino();
    let image = common::test_image();
    let base_config = Config {
        force_cpu: true,
        cache_dir: common::cache_dir("en"),
        openvino_dir: Some(ov_dir),
        ..Default::default()
    };
    let client = reqwest::Client::new();

    // One enabled model plus a disabled one that `default_model` points at (its file does not
    // even exist: disabled entries are never touched).
    let config = Config {
        default_model: Some("off-model".into()),
        models: vec![
            ModelConfig {
                name: Some("off-model".into()),
                path: common::models_dir().join("does-not-exist.onnx"),
                enabled: false,
                ..Default::default()
            },
            ModelConfig {
                path: model.clone(),
                family: ModelFamilyKind::Yolo5,
                ..Default::default()
            },
        ],
        ..base_config.clone()
    };
    let t0 = Instant::now();
    let run = Server::start_in_process(config).await;
    assert_eq!(run.registry().names(), ["IPcam-general"]);
    assert_eq!(
        run.registry()
            .default_model()
            .map(|w| w.name.clone())
            .as_deref(),
        Some("IPcam-general")
    );
    assert!(run.registry().by_name("off-model").is_none());

    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        let body = get_text(&client, &format!("{}/", run.base)).await;
        if body.contains("<td>Ready</td>") {
            assert!(!body.contains("off-model"), "{body}");
            break;
        }
        assert!(!body.contains("Failed:"), "model failed to load:\n{body}");
        assert!(
            Instant::now() < deadline,
            "model did not become ready in time"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    eprintln!("IPcam-general ready after {:?}", t0.elapsed());

    let r = post_image(
        &client,
        &format!("{}/v1/vision/detection", run.base),
        &image,
    )
    .await;
    assert!(r.success, "detection failed: {r:?}");
    assert!(!r.predictions.is_empty(), "no predictions: {r:?}");
    let r = post_image(
        &client,
        &format!("{}/v1/vision/custom/off-model", run.base),
        &image,
    )
    .await;
    assert!(!r.success);
    assert!(r.error.unwrap_or_default().contains("Unknown model"));
    let list: serde_json::Value = client
        .post(format!("{}/v1/vision/custom/list", run.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list["models"], serde_json::json!(["IPcam-general"]));

    // The config page still lists the disabled model, unchecked.
    let cfg = get_text(&client, &format!("{}/config", run.base)).await;
    assert!(
        cfg.contains("name=\"enabled\" value=\"off-model\" aria-label=\"Load off-model\">"),
        "{cfg}"
    );
    assert!(cfg.contains(
        "name=\"default_model\" value=\"IPcam-general\" aria-label=\"Default IPcam-general\" checked"
    ));
    run.stop().await;

    // Every model disabled: no workers, but the HTTP server and UI are up.
    let config = Config {
        models: vec![ModelConfig {
            path: model,
            family: ModelFamilyKind::Yolo5,
            enabled: false,
            ..Default::default()
        }],
        ..base_config
    };
    let run = Server::start_in_process(config).await;
    assert!(run.registry().names().is_empty());
    assert!(run.registry().default_model().is_none());
    let home = get_text(&client, &format!("{}/", run.base)).await;
    assert!(home.contains("No models are enabled"), "{home}");
    let cfg = get_text(&client, &format!("{}/config", run.base)).await;
    assert!(cfg.contains("action=\"/config/models#at-models\""));
    let r = post_image(
        &client,
        &format!("{}/v1/vision/detection", run.base),
        &image,
    )
    .await;
    assert!(!r.success);
    assert!(
        r.error
            .as_deref()
            .unwrap_or_default()
            .contains("enable a model"),
        "{r:?}"
    );
    run.stop().await;
}

// ---------------------------------------------------------------------------------------------
// In-process restart through the production run loop

/// Home page body, None while the server is down.
async fn home(client: &reqwest::Client, base: &str) -> Option<String> {
    client
        .get(format!("{base}/"))
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()
}

/// Wait until the running generation serves exactly `names` and all of them are Ready.
async fn wait_serving(client: &reqwest::Client, base: &str, names: &[&str]) {
    let expected: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    wait_until(
        &format!("{names:?} to be served"),
        Duration::from_secs(60),
        || async { list_models(client, base).await.as_ref() == Some(&expected) },
    )
    .await;
    wait_until(
        &format!("{names:?} to be ready"),
        Duration::from_secs(300),
        || async {
            let body = home(client, base).await.unwrap_or_default();
            assert!(!body.contains("Failed:"), "model failed to load:\n{body}");
            body.matches("<td>Ready</td>").count() == names.len()
        },
    )
    .await;
}

/// `requests_total` of `model` from `/prometheus` (None when the series is absent).
async fn prom_requests(client: &reqwest::Client, base: &str, model: &str) -> Option<u64> {
    let text = client
        .get(format!("{base}/prometheus"))
        .send()
        .await
        .expect("GET /prometheus")
        .text()
        .await
        .expect("prometheus text");
    let prefix = format!("blue_onyx_prism_requests_total{{model=\"{model}\"}} ");
    text.lines()
        .find_map(|l| l.strip_prefix(&prefix))
        .map(|v| v.parse().expect("numeric sample"))
}

fn base_config(ov_dir: &Path) -> Config {
    Config {
        force_cpu: true,
        cache_dir: common::cache_dir("en"),
        openvino_dir: Some(ov_dir.to_path_buf()),
        log_level: LogLevel::Warn,
        ..Default::default()
    }
}

fn labels(r: &VisionDetectionResponse) -> Vec<String> {
    r.predictions.iter().map(|p| p.label.clone()).collect()
}

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn switch_models_with_restart() {
    let ipcam = common::require_ipcam();
    let ov_dir = common::require_openvino();
    let (rtdetr, rt_family) = match common::optional_model("rt-detrv2-s.onnx") {
        Some(p) => (p, ModelFamilyKind::RtDetr),
        None => (ipcam.clone(), ModelFamilyKind::Yolo5),
    };
    let real_rtdetr = rt_family == ModelFamilyKind::RtDetr;
    let image = common::test_image();
    let dir = temp_dir("switch");
    let config_path = dir.join("cfg.json");
    let config = Config {
        models: vec![
            ModelConfig {
                name: Some("IPcam-general".into()),
                path: ipcam,
                family: ModelFamilyKind::Yolo5,
                ..Default::default()
            },
            ModelConfig {
                name: Some("rt-detrv2-s".into()),
                path: rtdetr,
                family: rt_family,
                enabled: false,
                ..Default::default()
            },
        ],
        ..base_config(&ov_dir)
    };
    let client = reqwest::Client::new();
    let t0 = Instant::now();
    let served = Server::start(config, config_path.clone()).await;
    let base = served.base.clone();
    wait_serving(&client, &base, &["IPcam-general"]).await;
    eprintln!(
        "generation 1 (IPcam-general) ready after {:?}",
        t0.elapsed()
    );
    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(r.success, "detection failed: {r:?}");
    // Counters are process-lifetime: they must survive the restarts below.
    let gen1_requests = prom_requests(&client, &base, "IPcam-general")
        .await
        .expect("IPcam-general series");
    assert!(gen1_requests >= 1);

    // Switch to rt-detrv2-s and restart in-process.
    let t1 = Instant::now();
    let page = post_form(
        &client,
        &format!("{base}/config/models"),
        "enabled=rt-detrv2-s&default_model=rt-detrv2-s&action=restart",
    )
    .await;
    assert!(page.contains("Restarting"), "{page}");
    wait_serving(&client, &base, &["rt-detrv2-s"]).await;
    eprintln!("generation 2 (rt-detrv2-s) ready after {:?}", t1.elapsed());

    let saved = Config::load(&config_path).unwrap();
    let flags: Vec<(String, bool)> = saved
        .models
        .iter()
        .map(|m| (m.effective_name(), m.enabled))
        .collect();
    assert_eq!(
        flags,
        [
            ("IPcam-general".to_string(), false),
            ("rt-detrv2-s".to_string(), true)
        ]
    );
    assert_eq!(saved.default_model.as_deref(), Some("rt-detrv2-s"));

    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(r.success, "detection failed: {r:?}");
    if real_rtdetr {
        assert!(labels(&r).iter().any(|l| l == "dog"), "no dog: {r:?}");
    }
    let r = post_image(
        &client,
        &format!("{base}/v1/vision/custom/IPcam-general"),
        &image,
    )
    .await;
    assert!(!r.success);
    assert!(
        r.error
            .as_deref()
            .unwrap_or_default()
            .contains("Unknown model"),
        "{r:?}"
    );

    // Switch back with "Save" only: applies at once, per model (IPcam-general starts,
    // rt-detrv2-s stops), without a new generation.
    let page = post_form(
        &client,
        &format!("{base}/config/models"),
        "enabled=IPcam-general&default_model=IPcam-general&action=save",
    )
    .await;
    assert!(page.contains("Loading IPcam-general on"), "{page}");
    assert!(page.contains("stopping rt-detrv2-s"), "{page}");
    assert!(!page.contains("Restart the server to apply"), "{page}");
    let saved = Config::load(&config_path).unwrap();
    assert!(saved.models[0].enabled && !saved.models[1].enabled);
    assert_eq!(saved.default_model.as_deref(), Some("IPcam-general"));
    assert_eq!(
        list_models(&client, &base).await.as_deref(),
        Some(&["IPcam-general".to_string()][..])
    );
    wait_serving(&client, &base, &["IPcam-general"]).await;
    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(r.success && !r.predictions.is_empty(), "{r:?}");
    let r = post_image(
        &client,
        &format!("{base}/v1/vision/custom/rt-detrv2-s"),
        &image,
    )
    .await;
    assert!(!r.success, "rt-detrv2-s was disabled: {r:?}");

    // ... until the restart. The listener stays up: requests issued during the restart get an
    // HTTP answer (from the old generation, or "initializing" from the new one), never a
    // connection error.
    let t2 = Instant::now();
    post_form(&client, &format!("{base}/config/restart"), "").await;
    let during = {
        let (client, base, image) = (client.clone(), base.clone(), image.clone());
        tokio::spawn(async move {
            let mut answers = Vec::new();
            let end = Instant::now() + Duration::from_secs(3);
            while Instant::now() < end {
                let part = reqwest::multipart::Part::bytes(image.clone())
                    .file_name("dog_bike_car.jpg")
                    .mime_str("image/jpeg")
                    .unwrap();
                let form = reqwest::multipart::Form::new().part("image", part);
                let r = client
                    .post(format!("{base}/v1/vision/detection"))
                    .multipart(form)
                    .send()
                    .await
                    .unwrap_or_else(|e| panic!("request during the restart failed: {e}"));
                assert!(r.status().is_success(), "HTTP {}", r.status());
                let body: VisionDetectionResponse = r.json().await.expect("detection JSON");
                answers.push(body);
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            answers
        })
    };
    wait_serving(&client, &base, &["IPcam-general"]).await;
    let answers = during.await.expect("requests during the restart");
    assert!(!answers.is_empty());
    for r in &answers {
        let e = r.error.as_deref().unwrap_or_default();
        assert!(
            r.success || e.contains("initializing") || e.contains("not running"),
            "unexpected answer during the restart: {r:?}"
        );
    }
    eprintln!(
        "{} requests during the restart: {} succeeded",
        answers.len(),
        answers.iter().filter(|r| r.success).count()
    );
    eprintln!(
        "generation 3 (IPcam-general) ready after {:?}",
        t2.elapsed()
    );
    let count_before = prom_requests(&client, &base, "IPcam-general")
        .await
        .expect("IPcam-general series");
    assert!(
        count_before >= gen1_requests,
        "request counter reset by the restarts: {count_before} < {gen1_requests}"
    );
    // The model that was switched off keeps its series.
    assert!(
        prom_requests(&client, &base, "rt-detrv2-s").await.is_some(),
        "series of the removed model disappeared"
    );
    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(r.success && !r.predictions.is_empty(), "{r:?}");
    let count_after = prom_requests(&client, &base, "IPcam-general")
        .await
        .expect("IPcam-general series");
    assert_eq!(count_after, count_before + 1);

    served.stop().await;
}

#[tokio::test]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn restart_into_zero_enabled_models_keeps_http_up() {
    let model = common::require_ipcam();
    let ov_dir = common::require_openvino();
    let image = common::test_image();
    let dir = temp_dir("zero");
    let config_path = dir.join("cfg.json");
    let config = Config {
        models: vec![ModelConfig {
            path: model,
            family: ModelFamilyKind::Yolo5,
            ..Default::default()
        }],
        ..base_config(&ov_dir)
    };
    let client = reqwest::Client::new();
    let served = Server::start(config.clone(), config_path.clone()).await;
    let base = served.base.clone();
    wait_serving(&client, &base, &["IPcam-general"]).await;

    // Edit the file on disk (every model disabled), then restart from the web UI.
    let mut off = config;
    off.models[0].enabled = false;
    off.save(&config_path).unwrap();
    let t0 = Instant::now();
    let page = post_form(&client, &format!("{base}/config/restart"), "").await;
    assert!(page.contains("Restarting"), "{page}");
    wait_until(
        "the zero-model generation",
        Duration::from_secs(60),
        || async {
            home(&client, &base)
                .await
                .is_some_and(|b| b.contains("No models are enabled"))
        },
    )
    .await;
    eprintln!("zero-model generation up after {:?}", t0.elapsed());

    assert_eq!(list_models(&client, &base).await, Some(Vec::new()));
    let r = post_image(&client, &format!("{base}/v1/vision/detection"), &image).await;
    assert!(!r.success, "{r:?}");
    let cfg = get_text(&client, &format!("{base}/config")).await;
    assert!(cfg.contains("action=\"/config/models#at-models\""), "{cfg}");
    assert!(cfg.contains("aria-label=\"Load IPcam-general\">"), "{cfg}");

    served.stop().await;
}
