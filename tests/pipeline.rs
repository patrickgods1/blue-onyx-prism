//! The pipelined worker under load (docs/REVIEW_PLAN.md, phase 2.4): 200 requests from 6
//! parallel clients through `/v1/vision/detection` all succeed, none is dropped, and every model
//! replies in the order it accepted the requests (checked with the worker's `accepted` /
//! `replied` trace events and sequence numbers in `image_name`). Then a burst with undecodable
//! images (answered as errors without disturbing the other slots) and a shutdown while
//! inferences are in flight.
//!
//! `#[ignore]`d: needs `models/IPcam-general.{onnx,yaml}` and `openvino/` under
//! `common::test_root()` (a missing one fails the test). The second model is `models/yolo26n.xml`
//! when present, else a second copy of IPcam-general.

mod common;

use blue_onyx_prism::api::VisionDetectionResponse;
use blue_onyx_prism::config::{Config, ModelConfig};
use blue_onyx_prism::metrics::DropReason;
use blue_onyx_prism::model::ModelFamilyKind;
use blue_onyx_prism::startup::ModelState;
use blue_onyx_prism::worker::ORDER_TARGET;
use common::Server;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::field::{Field, Visit};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;

const CLIENTS: usize = 6;
const REQUESTS: usize = 200;

/// One worker order event: (message, model, image).
type OrderEvent = (String, String, String);

#[derive(Default)]
struct Fields {
    message: String,
    model: String,
    image: String,
}

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let v = format!("{value:?}");
        match field.name() {
            "message" => self.message = v,
            "model" => self.model = v,
            "image" => self.image = v,
            _ => {}
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_debug(field, &format_args!("{value}"));
    }
}

/// Records the worker's order events, in the order the worker emitted them.
struct OrderLayer(Arc<Mutex<Vec<OrderEvent>>>);

impl<S: tracing::Subscriber> Layer<S> for OrderLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if event.metadata().target() != ORDER_TARGET {
            return;
        }
        let mut f = Fields::default();
        event.record(&mut f);
        self.0.lock().unwrap().push((f.message, f.model, f.image));
    }
}

async fn post(
    client: &reqwest::Client,
    url: &str,
    bytes: Vec<u8>,
    name: String,
) -> VisionDetectionResponse {
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(name)
        .mime_str("image/jpeg")
        .unwrap();
    let form = reqwest::multipart::Form::new().part("image", part);
    let resp = client.post(url).multipart(form).send().await.expect("POST");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    resp.json().await.expect("detection JSON")
}

/// Every model replied to exactly the requests it accepted, in the same order.
fn assert_fifo(events: &[OrderEvent], expect_total: usize) {
    let mut accepted: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut replied: HashMap<&str, Vec<&str>> = HashMap::new();
    for (msg, model, image) in events {
        match msg.as_str() {
            "accepted" => accepted.entry(model).or_default().push(image),
            "replied" => replied.entry(model).or_default().push(image),
            other => panic!("unexpected order event {other}"),
        }
    }
    let total: usize = accepted.values().map(Vec::len).sum();
    assert_eq!(total, expect_total, "accepted requests");
    for (model, acc) in &accepted {
        assert_eq!(
            replied.get(model),
            Some(acc),
            "model {model}: replies not in accept order"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs models/ and the runtime libraries; see tests/common"]
async fn pipelined_worker_keeps_order_under_load() {
    let general = common::require_ipcam();
    let ov_dir = common::require_openvino();
    // The second model: YOLO26n when exported, else another copy of the required model.
    let second = match common::optional_model("yolo26n.xml") {
        Some(path) => ModelConfig {
            path,
            ..Default::default()
        },
        None => ModelConfig {
            name: Some("IPcam-general-2".into()),
            path: general.clone(),
            family: ModelFamilyKind::Yolo5,
            ..Default::default()
        },
    };
    let events: Arc<Mutex<Vec<OrderEvent>>> = Arc::default();
    let subscriber = tracing_subscriber::registry().with(OrderLayer(events.clone()));
    tracing::subscriber::set_global_default(subscriber).expect("global subscriber");

    let image = common::test_image();
    // Two models on the CPU: the default one with two inferences in flight (exercises the slots
    // and the async path on any machine), the second with the profile's count.
    let config = Config {
        device: "openvino:cpu".into(),
        cache_dir: common::cache_dir("pipe"),
        openvino_dir: Some(ov_dir),
        preprocess_threads: 4,
        auto_download: false,
        models: vec![
            ModelConfig {
                path: general,
                family: ModelFamilyKind::Yolo5,
                inflight: Some(2),
                ..Default::default()
            },
            second,
        ],
        ..Default::default()
    };
    let server = Server::start_in_process(config).await;
    let (base, metrics) = (server.base.clone(), server.metrics.clone());
    let registry = server.registry().clone();

    let deadline = Instant::now() + Duration::from_secs(180);
    while !registry
        .workers()
        .iter()
        .all(|w| w.state.get() == ModelState::Ready)
    {
        for w in registry.workers() {
            if let ModelState::Failed(e) = w.state.get() {
                panic!("model {} failed to load: {e}", w.name);
            }
        }
        assert!(Instant::now() < deadline, "models did not become ready");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let names = registry.names();
    let detection = format!("{base}/v1/vision/detection");
    let custom = format!("{base}/v1/vision/custom/{}", names[1]);

    // 200 requests from 6 clients: each client sends its next request when the previous one
    // returned; every 5th goes to the second model.
    let client = reqwest::Client::new();
    let t = Instant::now();
    let tasks: Vec<_> = (0..CLIENTS)
        .map(|c| {
            let (client, image) = (client.clone(), image.clone());
            let (detection, custom) = (detection.clone(), custom.clone());
            tokio::spawn(async move {
                let mut out = Vec::new();
                for i in (c..REQUESTS).step_by(CLIENTS) {
                    let url = if i % 5 == 4 { &custom } else { &detection };
                    let r = post(&client, url, image.clone(), format!("seq-{i:04}.jpg")).await;
                    out.push((i, r));
                }
                out
            })
        })
        .collect();
    let mut responses = Vec::new();
    for t in tasks {
        responses.extend(t.await.expect("client task"));
    }
    let wall = t.elapsed();
    eprintln!(
        "{REQUESTS} requests from {CLIENTS} clients in {:.2} s ({:.1} requests/s)",
        wall.as_secs_f64(),
        REQUESTS as f64 / wall.as_secs_f64()
    );
    assert_eq!(responses.len(), REQUESTS);
    for (i, r) in &responses {
        assert!(r.success, "request {i} failed: {r:?}");
        assert!(!r.predictions.is_empty(), "request {i}: no predictions");
    }
    for m in metrics.models.read().unwrap().iter() {
        let drops: u64 = DropReason::ALL.iter().map(|r| m.dropped(*r)).sum();
        assert_eq!(drops, 0, "model {}: dropped requests", m.name);
    }
    assert_fifo(&events.lock().unwrap(), REQUESTS);

    // Undecodable images among good ones: errors for exactly those, the rest succeed.
    events.lock().unwrap().clear();
    let burst: Vec<_> = (0..24)
        .map(|i| {
            let bytes = if i % 4 == 1 {
                b"\xff\xd8 not a jpeg".to_vec()
            } else {
                image.clone()
            };
            let (client, url) = (client.clone(), detection.clone());
            tokio::spawn(async move {
                (
                    i,
                    post(&client, &url, bytes, format!("bad-{i:02}.jpg")).await,
                )
            })
        })
        .collect();
    for t in burst {
        let (i, r) = t.await.unwrap();
        assert_eq!(r.success, i % 4 != 1, "burst request {i}: {r:?}");
    }
    assert_fifo(&events.lock().unwrap(), 24);
    let t_shutdown = Instant::now();

    // Shutdown with inferences in flight: the workers cancel or finish them and exit.
    let inflight: Vec<_> = (0..12)
        .map(|i| {
            let (client, url, image) = (client.clone(), detection.clone(), image.clone());
            tokio::spawn(async move {
                let part =
                    reqwest::multipart::Part::bytes(image).file_name(format!("late-{i}.jpg"));
                let form = reqwest::multipart::Form::new().part("image", part);
                let _ = client.post(&url).multipart(form).send().await;
            })
        })
        .collect();
    tokio::time::sleep(Duration::from_millis(60)).await;
    server.cancel();
    drop(registry);
    server.stop().await;
    for t in inflight {
        let _ = t.await;
    }
    eprintln!("shutdown phase {:.1} s", t_shutdown.elapsed().as_secs_f64());
}
