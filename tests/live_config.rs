//! Config changes applied without a restart (docs/REVIEW_PLAN.md, phase 3), through the
//! production run loop (`runner::run_server_on`) and real OpenVINO:
//! - a global threshold saved with `POST /config` is used by the next detection;
//! - disabling a model on the Models card drops it from `/v1/vision/custom/list` at once and
//!   enabling it again loads and serves it;
//! - a per-model device change reloads only that model (`/stats.json` shows the reload, the
//!   generation in `/v1/config` stays the same).
//!
//! Skips (passes with a message) unless `models/IPcam-general.onnx` and `openvino/` exist under
//! the repo root (or `BLUE_ONYX_TEST_ROOT`).

use blue_onyx_prism::api::VisionDetectionResponse;
use blue_onyx_prism::cli::{self, LogReloadHandle};
use blue_onyx_prism::config::{Config, LogLevel, ModelConfig};
use blue_onyx_prism::model::ModelFamilyKind;
use blue_onyx_prism::runner;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

const TEST_IMAGE: &str = "tests/data/dog_bike_car.jpg";

fn repo_root() -> PathBuf {
    std::env::var_os("BLUE_ONYX_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("free port")
}

/// The model and OpenVINO, or None (skip) when they are not there.
fn fixtures(test: &str) -> Option<(PathBuf, PathBuf, Vec<u8>)> {
    let root = repo_root();
    let model = root.join("models").join("IPcam-general.onnx");
    let ov_dir = root.join("openvino");
    if !model.is_file() || !ov_dir.is_dir() {
        eprintln!(
            "skipping {test}: need {} and {}",
            model.display(),
            ov_dir.display()
        );
        return None;
    }
    let image = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_IMAGE))
        .expect("test image");
    Some((model, ov_dir, image))
}

/// The global subscriber can be installed once per test binary; every server shares the handle.
fn log_handle() -> LogReloadHandle {
    static LOG: OnceLock<LogReloadHandle> = OnceLock::new();
    LOG.get_or_init(|| cli::init_logging(LogLevel::Warn, None).expect("init logging"))
        .clone()
}

/// `runner::run_server_on` on its own thread and runtime, exactly like the binary.
struct Served {
    base: String,
    shutdown: CancellationToken,
    thread: std::thread::JoinHandle<anyhow::Result<()>>,
}

impl Served {
    fn start(config: Config, config_path: PathBuf) -> Self {
        config.save(&config_path).expect("write config");
        let base = format!("http://127.0.0.1:{}", config.port);
        let shutdown = CancellationToken::new();
        let token = shutdown.clone();
        let log = log_handle();
        let thread = std::thread::spawn(move || {
            let rt = runner::build_runtime()?;
            runner::run_server_on(&rt, config, config_path, log, token)
        });
        Self {
            base,
            shutdown,
            thread,
        }
    }

    fn stop(self) {
        self.shutdown.cancel();
        self.thread
            .join()
            .expect("server thread panicked")
            .expect("server result");
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bo_live_{tag}_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config_with(ov_dir: &Path, models: Vec<ModelConfig>) -> Config {
    Config {
        port: free_port(),
        cache_dir: std::env::temp_dir()
            .join(format!("bo_live_cache_{}", std::process::id()))
            .to_string_lossy()
            .to_string(),
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

/// Poll `check` every 50 ms until it returns true; panic after `timeout`.
async fn wait_until<F, Fut>(what: &str, timeout: Duration, mut check: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = Instant::now() + timeout;
    while !check().await {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn get_json(client: &reqwest::Client, url: &str) -> Option<serde_json::Value> {
    client.get(url).send().await.ok()?.json().await.ok()
}

/// `/v1/vision/custom/list` model names (None while the server is not up yet).
async fn list_models(client: &reqwest::Client, base: &str) -> Option<Vec<String>> {
    let v: serde_json::Value = client
        .post(format!("{base}/v1/vision/custom/list"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    serde_json::from_value(v["models"].clone()).ok()
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

async fn post_image(client: &reqwest::Client, url: &str, bytes: &[u8]) -> VisionDetectionResponse {
    let part = reqwest::multipart::Part::bytes(bytes.to_vec())
        .file_name("dog_bike_car.jpg")
        .mime_str("image/jpeg")
        .unwrap();
    let form = reqwest::multipart::Form::new().part("image", part);
    let resp = client.post(url).multipart(form).send().await.expect("POST");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    resp.json().await.expect("detection JSON")
}

/// POST a form as the Config page's script does (JSON answer).
async fn post_form(client: &reqwest::Client, url: &str, body: &str) -> serde_json::Value {
    let resp = client
        .post(url)
        .header("content-type", "application/x-www-form-urlencoded")
        .header("accept", "application/json")
        .body(body.to_string())
        .send()
        .await
        .expect("POST form");
    assert!(resp.status().is_success(), "{url}: HTTP {}", resp.status());
    let j: serde_json::Value = resp.json().await.expect("form JSON");
    assert_eq!(j["success"], true, "{j}");
    j
}

#[tokio::test]
async fn threshold_applies_to_the_next_detection() {
    let Some((model, ov_dir, image)) = fixtures("threshold_applies_to_the_next_detection") else {
        return;
    };
    let config = Config {
        confidence_threshold: 0.9,
        ..config_with(&ov_dir, vec![ipcam("IPcam-general", &model)])
    };
    let client = reqwest::Client::new();
    let served = Served::start(config, temp_dir("thr").join("cfg.json"));
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
    served.stop();
}

#[tokio::test]
async fn disable_and_enable_without_restart() {
    let Some((model, ov_dir, image)) = fixtures("disable_and_enable_without_restart") else {
        return;
    };
    let config = config_with(
        &ov_dir,
        vec![ipcam("ipcam-a", &model), ipcam("ipcam-b", &model)],
    );
    let client = reqwest::Client::new();
    let served = Served::start(config, temp_dir("enable").join("cfg.json"));
    let base = served.base.clone();
    wait_ready(&client, &base, &["ipcam-a", "ipcam-b"]).await;
    let gen0 = generation(&client, &base).await;

    // Disable ipcam-b: gone from the list within a second.
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
        Duration::from_secs(1),
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
    served.stop();
}

#[tokio::test]
async fn device_change_reloads_only_that_model() {
    let Some((model, ov_dir, image)) = fixtures("device_change_reloads_only_that_model") else {
        return;
    };
    let config = config_with(
        &ov_dir,
        vec![ipcam("ipcam-a", &model), ipcam("ipcam-b", &model)],
    );
    let client = reqwest::Client::new();
    let served = Served::start(config, temp_dir("device").join("cfg.json"));
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
    served.stop();
}
