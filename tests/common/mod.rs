//! Helpers shared by the integration tests (`mod common;` in each test binary).
//!
//! Fixture-dependent tests (real models + the OpenVINO / ONNX Runtime libraries) are
//! `#[ignore]`d so a plain `cargo test` stays hermetic; run them with
//! `cargo test -- --ignored --test-threads=1`. Inside, a missing *required* fixture is a hard
//! failure (see [`require_ipcam`], [`require_openvino`]), never a silent pass.
//!
//! Fixtures are looked up under [`test_root`]: `$BLUE_ONYX_TEST_ROOT`, else the repo root.
//!   - `<root>/models/IPcam-general.{onnx,yaml}` (required); `yolo26n/s.xml`, `rt-detrv2-s.onnx`
//!     are optional extras
//!   - `<root>/openvino/` (the OpenVINO runtime, `cargo run -- setup-openvino`)
//!   - `<root>/onnxruntime/` (`cargo run -- setup-onnxruntime`: flat with `--dir`, else flavor subdir + `active.txt`)
//!
//! Cargo compiles this module once per test binary, so not every helper is used everywhere.
#![allow(dead_code)]

use blue_onyx_prism::api::VisionDetectionResponse;
use blue_onyx_prism::cli::{self, LogReloadHandle};
use blue_onyx_prism::config::{Config, LogLevel};
use blue_onyx_prism::metrics::Metrics;
use blue_onyx_prism::registry::ModelRegistry;
use blue_onyx_prism::runner;
use blue_onyx_prism::server::{self, AppState};
use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

pub const IGNORE_REASON: &str = "needs models/ and the runtime libraries; see tests/common";
pub const TEST_IMAGE: &str = "tests/data/dog_bike_car.jpg";

// ---------------------------------------------------------------------------------------------
// Fixture locations

/// The checkout the tests were built from (for files that ship in the repo).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Where `models/`, `openvino/` and `onnxruntime/` live: `$BLUE_ONYX_TEST_ROOT`, else the repo
/// root. (CI points it at `target/release`, where the setup commands install everything.)
pub fn test_root() -> PathBuf {
    std::env::var_os("BLUE_ONYX_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(repo_root)
}

pub fn models_dir() -> PathBuf {
    test_root().join("models")
}

/// The OpenVINO runtime folder. Panics when it is missing.
pub fn require_openvino() -> PathBuf {
    let dir = test_root().join("openvino");
    assert!(
        dir.is_dir(),
        "required fixture missing: OpenVINO runtime folder {} (run `cargo run -- setup-openvino` \
         or set BLUE_ONYX_TEST_ROOT); these tests are #[ignore]d unless fixtures are installed",
        dir.display()
    );
    dir
}

/// `models/IPcam-general.onnx` (its `.yaml` must sit next to it). Panics when missing.
pub fn require_ipcam() -> PathBuf {
    let onnx = models_dir().join("IPcam-general.onnx");
    let yaml = models_dir().join("IPcam-general.yaml");
    for f in [&onnx, &yaml] {
        assert!(
            f.is_file(),
            "required fixture missing: {} (run `cargo run -- download-models --name IPcam-general` \
             or set BLUE_ONYX_TEST_ROOT)",
            f.display()
        );
    }
    onnx
}

/// An optional extra model under `models/`; None (with a note) when absent.
pub fn optional_model(file: &str) -> Option<PathBuf> {
    let p = models_dir().join(file);
    if p.is_file() {
        Some(p)
    } else {
        eprintln!(
            "optional model {} not present: skipping that part",
            p.display()
        );
        None
    }
}

/// The JPEG every detection test posts.
pub fn test_image() -> Vec<u8> {
    std::fs::read(repo_root().join(TEST_IMAGE)).expect("test image")
}

// ---------------------------------------------------------------------------------------------
// Ports, temp folders, logging

/// A free loopback port, never handed out twice within this process.
pub fn free_port() -> u16 {
    static GIVEN: Mutex<Option<HashSet<u16>>> = Mutex::new(None);
    loop {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .map(|a| a.port())
            .expect("free port");
        let mut given = GIVEN.lock().unwrap_or_else(|e| e.into_inner());
        if given.get_or_insert_with(HashSet::new).insert(port) {
            return port;
        }
    }
}

/// A fresh, empty folder under the system temp dir.
pub fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("bo_test_{tag}_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// A fresh compile-cache folder (string form for `Config::cache_dir`). Tests do not share one,
/// so parallel tests never write the same cache entries.
pub fn cache_dir(tag: &str) -> String {
    temp_dir(&format!("cache_{tag}"))
        .to_string_lossy()
        .into_owned()
}

/// The global subscriber can be installed once per test binary; every server shares the handle.
pub fn log_handle() -> LogReloadHandle {
    static LOG: OnceLock<LogReloadHandle> = OnceLock::new();
    LOG.get_or_init(|| cli::init_logging(LogLevel::Warn, None).expect("init logging"))
        .clone()
}

// ---------------------------------------------------------------------------------------------
// HTTP helpers

/// Poll `check` every 50 ms until it returns true; panic after `timeout`.
pub async fn wait_until<F, Fut>(what: &str, timeout: Duration, mut check: F)
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

/// POST `bytes` as the multipart `image` field (no other fields).
pub async fn post_image(
    client: &reqwest::Client,
    url: &str,
    bytes: &[u8],
) -> VisionDetectionResponse {
    post_image_with(client, url, bytes, None).await
}

/// POST `bytes` as the multipart `image` field, plus `min_confidence` when given.
pub async fn post_image_with(
    client: &reqwest::Client,
    url: &str,
    bytes: &[u8],
    min_confidence: Option<&str>,
) -> VisionDetectionResponse {
    let part = reqwest::multipart::Part::bytes(bytes.to_vec())
        .file_name("dog_bike_car.jpg")
        .mime_str("image/jpeg")
        .unwrap();
    let mut form = reqwest::multipart::Form::new().part("image", part);
    if let Some(c) = min_confidence {
        form = form.text("min_confidence", c.to_string());
    }
    let resp = client.post(url).multipart(form).send().await.expect("POST");
    assert!(resp.status().is_success(), "{url}: HTTP {}", resp.status());
    resp.json().await.expect("detection JSON")
}

/// POST an urlencoded form; the HTML answer.
pub async fn post_form(client: &reqwest::Client, url: &str, body: &str) -> String {
    let resp = client
        .post(url)
        .header("content-type", "application/x-www-form-urlencoded")
        .body(body.to_string())
        .send()
        .await
        .expect("POST form");
    assert!(resp.status().is_success(), "{url}: HTTP {}", resp.status());
    resp.text().await.expect("form response")
}

/// POST a form as the Config page's script does (JSON answer, `success` must be true).
pub async fn post_form_json(client: &reqwest::Client, url: &str, body: &str) -> serde_json::Value {
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

pub async fn get_json(client: &reqwest::Client, url: &str) -> Option<serde_json::Value> {
    client.get(url).send().await.ok()?.json().await.ok()
}

/// `/v1/vision/custom/list` model names, None while the server is down (between generations).
pub async fn list_models(client: &reqwest::Client, base: &str) -> Option<Vec<String>> {
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

// ---------------------------------------------------------------------------------------------
// Server harness

enum Mode {
    /// Registry + `server::serve_on` driven by the test.
    InProcess {
        server: tokio::task::JoinHandle<anyhow::Result<()>>,
        registry: Arc<ModelRegistry>,
    },
    /// `runner::run_server_on` (registry + HTTP generations, restart) on its own thread, as the
    /// binary and the Windows service run it.
    Runner {
        thread: std::thread::JoinHandle<anyhow::Result<()>>,
    },
}

/// A running Blue Onyx Prism server on a free loopback port.
pub struct Server {
    pub base: String,
    pub token: CancellationToken,
    /// Metrics of the in-process server (a fresh, unused instance in runner mode).
    pub metrics: Arc<Metrics>,
    mode: Mode,
}

impl Server {
    /// Start the production run loop (`runner::run_server_on`) with `config` saved to
    /// `config_path`. `config.port` is replaced by a free port. Returns once HTTP answers.
    pub async fn start(config: Config, config_path: PathBuf) -> Self {
        // A port picked by `free_port` can be taken by another process before the server binds
        // it; when the run loop exits early (bind error), retry on a new port.
        let mut last_err = String::new();
        for _ in 0..5 {
            let mut config = config.clone();
            config.port = free_port();
            config.save(&config_path).expect("write config");
            let base = format!("http://127.0.0.1:{}", config.port);
            let token = CancellationToken::new();
            let shutdown = token.clone();
            let log = log_handle();
            let path = config_path.clone();
            let thread = std::thread::spawn(move || {
                let rt = runner::build_runtime()?;
                runner::run_server_on(&rt, config, path, log, shutdown)
            });
            let server = Self {
                base,
                token,
                metrics: Arc::new(Metrics::new(blue_onyx_prism::VERSION)),
                mode: Mode::Runner { thread },
            };
            match server.wait_http_or_exit().await {
                Ok(()) => return server,
                Err(e) => last_err = e,
            }
        }
        panic!("server did not start after 5 attempts: {last_err}");
    }

    /// Start a registry and the HTTP server directly (no restart support), exposing the
    /// registry and metrics. Returns once the listener is bound.
    pub async fn start_in_process(config: Config) -> Self {
        let token = CancellationToken::new();
        let metrics = Arc::new(Metrics::new(blue_onyx_prism::VERSION));
        let registry =
            Arc::new(ModelRegistry::start(&config, &metrics, token.clone()).expect("registry"));
        let config_path = temp_dir("config").join("config.json");
        let state = Arc::new(AppState::new(
            registry.clone(),
            metrics.clone(),
            config,
            config_path,
        ));
        let port = free_port();
        let listeners = server::bind([127, 0, 0, 1].into(), port)
            .await
            .expect("bind");
        let server = tokio::spawn(server::serve_on(listeners, state.into(), token.clone()));
        Self {
            base: format!("http://127.0.0.1:{port}"),
            token,
            metrics,
            mode: Mode::InProcess { server, registry },
        }
    }

    /// The registry of an in-process server.
    pub fn registry(&self) -> &Arc<ModelRegistry> {
        match &self.mode {
            Mode::InProcess { registry, .. } => registry,
            Mode::Runner { .. } => panic!("registry() needs Server::start_in_process"),
        }
    }

    /// Ask the server to shut down without waiting (see [`Server::stop`]).
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Wait until HTTP answers; Err (with the run loop's error) when the loop exited first.
    async fn wait_http_or_exit(&self) -> Result<(), String> {
        let client = reqwest::Client::new();
        let url = format!("{}/", self.base);
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if client.get(&url).send().await.is_ok() {
                return Ok(());
            }
            if let Mode::Runner { thread } = &self.mode
                && thread.is_finished()
            {
                return Err(format!("run loop exited before serving {}", self.base));
            }
            assert!(Instant::now() < deadline, "HTTP server did not answer");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Shut down and wait for everything to stop: the HTTP tasks, then the workers (30 s at
    /// most, panics otherwise). In-flight handlers may still hold the registry for a moment
    /// after the listener is gone, so this waits for the last reference instead of assuming it.
    pub async fn stop(self) {
        self.token.cancel();
        match self.mode {
            Mode::Runner { thread } => {
                thread
                    .join()
                    .expect("server thread panicked")
                    .expect("server result");
            }
            Mode::InProcess { server, registry } => {
                server.await.unwrap().unwrap();
                let deadline = Instant::now() + Duration::from_secs(30);
                let mut registry = registry;
                let reg = loop {
                    match Arc::try_unwrap(registry) {
                        Ok(reg) => break reg,
                        Err(shared) => {
                            assert!(
                                Instant::now() < deadline,
                                "registry still shared 30 s after the server stopped \
                                 ({} references)",
                                Arc::strong_count(&shared)
                            );
                            registry = shared;
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }
                    }
                };
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    reg.shutdown();
                    let _ = tx.send(());
                });
                rx.recv_timeout(Duration::from_secs(30))
                    .expect("workers did not stop after the shutdown token was cancelled");
            }
        }
    }
}
