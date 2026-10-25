//! In-process tests of the run loop (PL-12): a real worker thread, spawned like the registry
//! does (`spawn_worker`, bounded crossbeam channel, shared `PrepPool`), on the scripted
//! `backend::fake` runtime. They cover FIFO replies with several slots, a prep failure in the
//! middle, the timeout before submit, shutdown with a busy slot, and the stuck-inference path.

use super::watch::STUCK_PREFIX;
use super::*;
use crate::backend::fake::{Call, FakeScript, Step, yolo26_output};
use crate::image::RgbImage;
use crate::metrics::RouteFallback;
use crate::model::ModelFamilyKind;
use tokio::sync::oneshot;

/// Model input (width, height) of the fake model.
const INPUT: (u32, u32) = (64, 64);
/// Generous bound for anything that should happen "at once" (CI runners are slow).
const SOON: Duration = Duration::from_secs(10);
/// Worker queue length. Tests submit with `try_send` (as the server does) and send up to
/// `MAX_CONSECUTIVE_DEGENERATE + 1` requests before waiting, so the queue must hold a whole
/// burst: with 16, a slow runner whose worker had not drained 10 of 26 yet got `Full`.
const QUEUE: usize = 64;
const _: () = assert!(QUEUE > super::run::MAX_CONSECUTIVE_DEGENERATE as usize + 1);

fn classes() -> Vec<String> {
    (0..5).map(|i| format!("c{i}")).collect()
}

/// A 64x48 JPEG (letterboxed into the 64x64 input with 8 px of padding above and below).
fn jpeg() -> Vec<u8> {
    let (w, h) = (64u32, 48u32);
    let img = RgbImage {
        width: w,
        height: h,
        rgb: (0..w * h * 3).map(|i| (i % 251) as u8).collect(),
    };
    crate::image::encode_jpeg(&img, 90).unwrap()
}

/// One detection of class `class` (score 0.9) in model-input pixels.
fn det(class: usize, delay_ms: u64) -> Step {
    Step::detections(
        &[[10.0, 18.0, 30.0, 38.0, 0.9, class as f32]],
        Duration::from_millis(delay_ms),
    )
}

struct Harness {
    /// Shared so a [`Peers`] directory can point at it (weakly).
    handle: Arc<WorkerHandle>,
    script: FakeScript,
    shutdown: CancellationToken,
}

impl Harness {
    /// Spawn a worker on the fake runtime and wait until its model is ready.
    fn start(slots: usize, request_timeout: Duration) -> Self {
        let live = LiveSettings {
            request_timeout,
            ..LiveSettings::default()
        };
        Self::start_named("fake", slots, live, Peers::new())
    }

    /// [`Self::start`] for model `name` with `live` settings and the worker directory `peers`.
    fn start_named(name: &str, slots: usize, live: LiveSettings, peers: Peers) -> Self {
        let script = FakeScript::new(slots, INPUT);
        let runtimes = Arc::new(Mutex::new(Runtimes::fake(script.clone())));
        let path = PathBuf::from("fake.onnx");
        let cfg = WorkerConfig {
            name: name.into(),
            model: ModelConfig {
                path: path.clone(),
                family: ModelFamilyKind::Yolo26,
                ..ModelConfig::default()
            },
            device: DeviceSpec::Auto,
            load: LoadRequest {
                path,
                requested: "auto".into(),
                gpu_precision: None,
            },
            classes: classes(),
            live: live.shared(),
            queue_size: QUEUE,
            lazy: false,
            can_use_gpu: false,
            wait: None,
            concurrency: None,
            prep: Some(PrepPool::new(2)),
            peers,
        };
        let metrics = Arc::new(ModelMetrics::new(name, "auto", ""));
        let shutdown = CancellationToken::new();
        let handle = Arc::new(spawn_worker(runtimes, cfg, metrics, shutdown.clone()));
        let h = Self {
            handle,
            script,
            shutdown,
        };
        h.wait_for("the model to load", || {
            h.handle.state.get() == ModelState::Ready
        });
        h
    }

    fn send_at(&self, image: Vec<u8>, name: &str, enqueued: Instant) -> Reply {
        let (tx, rx) = oneshot::channel();
        let req = VisionDetectionRequest {
            min_confidence: 0.0,
            image_data: image.into(),
            image_name: name.into(),
        };
        self.handle
            .sender
            .try_send(WorkItem::Detect(req, tx, enqueued))
            .unwrap();
        Reply(rx)
    }

    fn send(&self, name: &str) -> Reply {
        self.send_at(jpeg(), name, Instant::now())
    }

    fn wait_for(&self, what: &str, cond: impl Fn() -> bool) {
        let t = Instant::now();
        while !cond() {
            assert!(t.elapsed() < SOON, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Cancel and join; returns how long the worker took to exit after the cancel.
    fn stop(self) -> Duration {
        let t = Instant::now();
        self.shutdown.cancel();
        match Arc::try_unwrap(self.handle) {
            Ok(h) => h.join(),
            Err(_) => panic!("a test still holds the worker handle"),
        }
        t.elapsed()
    }
}

struct Reply(oneshot::Receiver<VisionDetectionResponse>);

impl Reply {
    /// The reply, or None when it has not arrived yet.
    fn poll(&mut self) -> Option<VisionDetectionResponse> {
        match self.0.try_recv() {
            Ok(r) => Some(r),
            Err(oneshot::error::TryRecvError::Empty) => None,
            Err(oneshot::error::TryRecvError::Closed) => panic!("the worker dropped the reply"),
        }
    }

    fn wait(mut self) -> VisionDetectionResponse {
        let t = Instant::now();
        loop {
            if let Some(r) = self.poll() {
                return r;
            }
            assert!(t.elapsed() < SOON, "no reply");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Poll every reply until all arrived; each with the instant it was first seen.
fn wait_all(mut replies: Vec<Reply>) -> Vec<(VisionDetectionResponse, Instant)> {
    let mut got: Vec<Option<(VisionDetectionResponse, Instant)>> =
        replies.iter().map(|_| None).collect();
    let t = Instant::now();
    while got.iter().any(Option::is_none) {
        assert!(t.elapsed() < SOON, "missing replies");
        for (r, g) in replies.iter_mut().zip(got.iter_mut()) {
            if g.is_none()
                && let Some(resp) = r.poll()
            {
                *g = Some((resp, Instant::now()));
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    got.into_iter().map(Option::unwrap).collect()
}

fn label(r: &VisionDetectionResponse) -> &str {
    assert!(r.success, "{}", r.message);
    assert_eq!(r.predictions.len(), 1, "{r:?}");
    &r.predictions[0].label
}

#[test]
fn replies_keep_request_order_across_slots() {
    let h = Harness::start(2, Duration::from_secs(30));
    // Later requests finish first (slot 1 at once, slot 0 after 150 ms): replies must still
    // come in request order, each with its own detections.
    h.script
        .push([det(0, 150), det(1, 0), det(2, 60), det(3, 0), det(4, 20)]);
    let started = Instant::now();
    let replies: Vec<Reply> = (0..5).map(|i| h.send(&format!("img{i}.jpg"))).collect();
    let got = wait_all(replies);
    for (i, (r, _)) in got.iter().enumerate() {
        assert_eq!(label(r), format!("c{i}"), "request {i}");
        assert_eq!(r.executionProvider, "OpenVINO CPU (Fake CPU)");
    }
    for w in got.windows(2) {
        assert!(w[0].1 <= w[1].1, "replies out of order");
    }
    // Request 1 finished at once but waited for request 0.
    assert!(got[1].1 - started >= Duration::from_millis(140));
    // Two slots were used, the request submits are in order and the warm-up covered slot 1.
    let calls = h.script.calls();
    assert_eq!(calls[0], Call::WarmupSubmit { slot: 0 });
    assert_eq!(calls[1], Call::Collect { slot: 0 });
    assert_eq!(calls[2], Call::WarmupSubmit { slot: 1 });
    let submits: Vec<(usize, u64)> = calls
        .iter()
        .filter_map(|c| match c {
            Call::Submit { slot, seq } => Some((*slot, *seq)),
            _ => None,
        })
        .collect();
    assert_eq!(submits.len(), 5);
    assert!(submits.windows(2).all(|w| w[0].1 < w[1].1));
    assert!(submits.iter().any(|s| s.0 == 0) && submits.iter().any(|s| s.0 == 1));
    assert!(h.stop() < SOON);
}

#[test]
fn prep_failure_in_the_middle_answers_only_that_request() {
    let h = Harness::start(1, Duration::from_secs(30));
    h.script.push([det(0, 0), det(1, 0), det(3, 0), det(4, 0)]);
    let replies: Vec<Reply> = (0..5)
        .map(|i| {
            let image = if i == 2 {
                b"definitely not an image".to_vec()
            } else {
                jpeg()
            };
            h.send_at(image, &format!("img{i}.jpg"), Instant::now())
        })
        .collect();
    let got = wait_all(replies);
    for (i, (r, _)) in got.iter().enumerate() {
        if i == 2 {
            assert!(!r.success);
            assert!(r.error.is_some(), "{r:?}");
            assert_eq!(r.executionProvider, "OpenVINO CPU (Fake CPU)");
        } else {
            assert_eq!(label(r), format!("c{i}"), "request {i}");
        }
    }
    for w in got.windows(2) {
        assert!(w[0].1 <= w[1].1, "replies out of order");
    }
    assert_eq!(h.script.submits(), 4, "the bad image is never submitted");
    assert_eq!(h.script.remaining(), 0);
    h.stop();
}

#[test]
fn request_timed_out_before_submit_is_never_submitted() {
    let h = Harness::start(1, Duration::from_secs(2));
    h.script.push([det(0, 0), det(2, 0)]);
    let old = Instant::now()
        .checked_sub(Duration::from_secs(5))
        .unwrap_or_else(Instant::now);
    let a = h.send("a.jpg");
    let b = h.send_at(jpeg(), "b.jpg", old);
    let c = h.send("c.jpg");
    let got = wait_all(vec![a, b, c]);
    assert_eq!(label(&got[0].0), "c0");
    let dropped = &got[1].0;
    assert!(!dropped.success);
    assert!(
        dropped.message.starts_with("Request dropped: waited"),
        "{}",
        dropped.message
    );
    assert!(
        dropped.message.contains("(timeout 2000 ms)"),
        "{}",
        dropped.message
    );
    assert_eq!(label(&got[2].0), "c2");
    assert_eq!(h.script.submits(), 2);
    h.stop();
}

#[test]
fn shutdown_with_a_busy_slot_answers_everything_and_exits() {
    let h = Harness::start(1, Duration::from_secs(30));
    h.script.push([Step::Hang]);
    let first = h.send("hung.jpg");
    h.wait_for("the hung inference to start", || h.script.submits() == 1);
    // Three wait in the pipeline (slots + prep threads), the rest stay in the channel.
    let queued: Vec<Reply> = (0..5).map(|i| h.send(&format!("q{i}.jpg"))).collect();
    std::thread::sleep(Duration::from_millis(50));
    let script = h.script.clone();
    let exit = h.stop();
    // The loop re-checks the token at least every POLL; nothing waits for the hung inference.
    assert!(
        exit < POLL + Duration::from_secs(2),
        "worker took {exit:?} to exit"
    );
    let mut all = vec![first];
    all.extend(queued);
    for (r, _) in wait_all(all) {
        assert!(!r.success);
        assert_eq!(r.message, "Model 'fake' is shutting down");
    }
    let calls = script.calls();
    assert!(calls.contains(&Call::Abort), "{calls:?}");
    // The only collect of slot 0 is the warm-up's; the hung request was never collected.
    let after_request = calls
        .iter()
        .position(|c| matches!(c, Call::Submit { .. }))
        .unwrap();
    assert!(
        !calls[after_request..].contains(&Call::Collect { slot: 0 }),
        "{calls:?}"
    );
    assert_eq!(script.submits(), 1);
}

/// A hung inference outlasts the request timeout twice: the model turns `Failed` with the
/// stuck message while the worker is still blocked.
fn hang_until_stuck(h: &Harness) -> Reply {
    h.script.push([Step::Hang]);
    let a = h.send("hung.jpg");
    h.wait_for("the model to be marked stuck", || h.handle.is_stuck());
    let ModelState::Failed(msg) = h.handle.state.get() else {
        panic!("expected Failed, got {:?}", h.handle.state.get());
    };
    assert!(msg.starts_with(STUCK_PREFIX), "{msg}");
    assert!(msg.contains("openvino:cpu"), "{msg}");
    a
}

#[test]
fn stuck_inference_fails_the_model_and_later_requests_fail_fast() {
    let h = Harness::start(1, Duration::from_millis(300));
    let a = hang_until_stuck(&h);
    let behind = h.send("behind.jpg");
    // The runtime finally gives up with an error: the model stays failed.
    h.script.release(Err("device lost".into()));
    let ra = a.wait();
    assert!(!ra.success);
    assert!(ra.message.contains("device lost"), "{}", ra.message);
    let rb = behind.wait();
    assert!(!rb.success);
    assert!(
        rb.message
            .starts_with("Model 'fake' is unavailable: inference stuck on"),
        "{}",
        rb.message
    );
    assert!(h.handle.is_stuck());
    // From now on the worker answers at once (the server answers 503 before even queuing).
    let t = Instant::now();
    let rc = h.send("later.jpg").wait();
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    assert!(!rc.success);
    assert!(
        rc.message.contains("is unavailable: inference stuck"),
        "{}",
        rc.message
    );
    let submits = h.script.submits();
    h.send("later2.jpg").wait();
    assert_eq!(h.script.submits(), submits, "the parked engine is not used");
    h.stop();
}

#[test]
fn stuck_inference_that_finishes_makes_the_model_ready_again() {
    let h = Harness::start(1, Duration::from_millis(300));
    let a = hang_until_stuck(&h);
    h.script.release(Ok(vec![yolo26_output(&[[
        10.0, 18.0, 30.0, 38.0, 0.9, 3.0,
    ]])]));
    assert_eq!(label(&a.wait()), "c3");
    h.wait_for("the model to be ready again", || {
        h.handle.state.get() == ModelState::Ready
    });
    assert!(!h.handle.is_stuck());
    h.script.push([det(1, 0)]);
    assert_eq!(label(&h.send("next.jpg").wait()), "c1");
    h.stop();
}

#[test]
fn dead_session_fails_the_model() {
    let h = Harness::start(1, Duration::from_secs(30));
    h.script
        .push([Step::Panic, Step::Fail("not reached".into())]);
    let r = h.send("a.jpg").wait();
    assert!(!r.success);
    assert!(r.message.contains("panicked"), "{}", r.message);
    h.wait_for("the model to fail", || h.handle.is_stuck());
    let ModelState::Failed(msg) = h.handle.state.get() else {
        panic!("expected Failed");
    };
    assert!(
        msg.starts_with(STUCK_PREFIX) && msg.contains("runtime error"),
        "{msg}"
    );
    let r = h.send("b.jpg").wait();
    assert!(r.message.contains("is unavailable"), "{}", r.message);
    assert_eq!(h.script.submits(), 1);
    h.stop();
}

/// A streak of inference errors (a reset GPU fails every `run`) marks the model broken, so
/// the supervisor can reload it instead of every request failing for good.
#[test]
fn repeated_inference_errors_fail_the_model() {
    use super::run::MAX_CONSECUTIVE_ERRORS;
    use super::watch::BROKEN_PREFIX;
    let h = Harness::start(1, Duration::from_secs(30));
    let n = MAX_CONSECUTIVE_ERRORS as usize;
    // One error short of the limit, then a success: still ready, streak reset.
    h.script
        .push(std::iter::repeat_n(Step::Fail("CUDA failure 700".into()), n - 1).chain([det(2, 0)]));
    let got = wait_all((0..n).map(|i| h.send(&format!("a{i}.jpg"))).collect());
    assert!(got[..n - 1].iter().all(|(r, _)| !r.success));
    assert_eq!(label(&got[n - 1].0), "c2");
    assert_eq!(h.handle.state.get(), ModelState::Ready);
    assert!(!h.handle.is_stuck());
    // The full streak: broken.
    h.script.push(std::iter::repeat_n(
        Step::Fail("CUDA failure 700".into()),
        n,
    ));
    let got = wait_all((0..n).map(|i| h.send(&format!("b{i}.jpg"))).collect());
    assert!(got.iter().all(|(r, _)| !r.success));
    h.wait_for("the model to be marked broken", || h.handle.is_stuck());
    let ModelState::Failed(msg) = h.handle.state.get() else {
        panic!("expected Failed, got {:?}", h.handle.state.get());
    };
    assert!(msg.starts_with(BROKEN_PREFIX), "{msg}");
    assert!(msg.contains("CUDA failure 700"), "{msg}");
    assert!(msg.contains("reloaded automatically"), "{msg}");
    let r = h.send("later.jpg").wait();
    assert!(
        r.message.contains("is unavailable: inference broken on"),
        "{}",
        r.message
    );
    assert_eq!(h.script.submits(), 2 * n, "the parked engine is not used");
    h.stop();
}

/// A reset device can answer every inference with unwritten (all-zero) buffers and no error:
/// enough of those in a row mark the model broken too.
#[test]
fn all_zero_outputs_fail_the_model() {
    use super::run::MAX_CONSECUTIVE_DEGENERATE;
    use super::watch::BROKEN_PREFIX;
    let h = Harness::start(1, Duration::from_secs(30));
    let n = MAX_CONSECUTIVE_DEGENERATE as usize;
    let zeros = || Step::Ok(vec![yolo26_output(&[[0.0; 6]; 3])], Duration::ZERO);
    // Empty results are not degenerate; a real detection resets the streak.
    h.script
        .push(std::iter::repeat_with(zeros).take(n - 1).chain([
            Step::Ok(vec![yolo26_output(&[])], Duration::ZERO),
            det(1, 0),
        ]));
    let got = wait_all((0..=n).map(|i| h.send(&format!("a{i}.jpg"))).collect());
    assert!(got.iter().all(|(r, _)| r.success));
    assert_eq!(h.handle.state.get(), ModelState::Ready);
    h.script.push(std::iter::repeat_with(zeros).take(n));
    let got = wait_all((0..n).map(|i| h.send(&format!("b{i}.jpg"))).collect());
    assert!(got.iter().all(|(r, _)| r.success && r.count == 0));
    h.wait_for("the model to be marked broken", || h.handle.is_stuck());
    let ModelState::Failed(msg) = h.handle.state.get() else {
        panic!("expected Failed");
    };
    assert!(
        msg.starts_with(BROKEN_PREFIX) && msg.contains("only zeros"),
        "{msg}"
    );
    h.stop();
}

#[test]
fn per_request_failure_keeps_the_model_ready() {
    let h = Harness::start(1, Duration::from_secs(30));
    h.script
        .push([Step::Fail("inference failed".into()), det(4, 0)]);
    let a = h.send("a.jpg");
    let b = h.send("b.jpg");
    let got = wait_all(vec![a, b]);
    assert!(!got[0].0.success);
    assert!(
        got[0].0.message.contains("inference failed"),
        "{}",
        got[0].0.message
    );
    assert_eq!(label(&got[1].0), "c4");
    assert_eq!(h.handle.state.get(), ModelState::Ready);
    h.stop();
}

/// A 64x48 gray JPEG (an IR frame for [`crate::image_stats`]).
fn gray_jpeg() -> Vec<u8> {
    let (w, h) = (64u32, 48u32);
    let img = RgbImage {
        width: w,
        height: h,
        rgb: (0..w * h).flat_map(|i| [(i % 200) as u8 + 30; 3]).collect(),
    };
    crate::image::encode_jpeg(&img, 90).unwrap()
}

#[test]
fn night_rules_and_static_memory_apply_in_the_worker() {
    use crate::model::filters::{FilterRules, SuppressReason};
    let h = Harness::start(1, Duration::from_secs(30));
    let mut live = LiveSettings {
        request_timeout: Duration::from_secs(30),
        ..LiveSettings::default()
    };
    live.filters = FilterRules {
        night_confidence_threshold: Some(0.95),
        label_thresholds: [("C1".to_string(), 0.95)].into_iter().collect(),
        ..FilterRules::default()
    };
    live.static_suppression = crate::static_memory::StaticSuppression {
        enabled: true,
        hits: 2,
        labels: vec!["c2".into()],
        ..Default::default()
    };
    assert!(h.handle.set_live(live));
    // Day frame: c0 (0.9) kept; c1 under its label threshold.
    h.script.push([det(0, 0), det(1, 0)]);
    assert_eq!(label(&h.send("a.jpg").wait()), "c0");
    let r = h.send("a.jpg").wait();
    assert!(r.success && r.predictions.is_empty(), "{r:?}");
    // IR frame: the night threshold removes c0.
    h.script.push([det(0, 0)]);
    let r = h.send_at(gray_jpeg(), "a.jpg", Instant::now()).wait();
    assert!(r.success && r.predictions.is_empty(), "{r:?}");
    // The same c2 box from one camera: reported twice, then static.
    h.script.push((0..4).map(|_| det(2, 0)));
    let counts: Vec<usize> = (0..4)
        .map(|_| {
            h.send("Driveway.20240101_1200.jpg")
                .wait()
                .predictions
                .len()
        })
        .collect();
    assert_eq!(counts, [1, 1, 0, 0]);
    let m = &h.handle.metrics;
    assert_eq!(m.suppressed(SuppressReason::LabelThreshold), 1);
    assert_eq!(m.suppressed(SuppressReason::NightThreshold), 1);
    assert_eq!(m.suppressed(SuppressReason::Static), 2);
    assert!(h.stop() < SOON);
}

#[test]
fn night_threshold_runs_the_same_model_with_another_threshold() {
    let h = Harness::start(1, Duration::from_secs(30));
    let live = LiveSettings {
        request_timeout: Duration::from_secs(30),
        confidence_threshold: 0.95,
        night_threshold: Some(0.5),
        ..LiveSettings::default()
    };
    assert!(h.handle.set_live(live));
    // Day frame: the 0.9 box is under the day threshold.
    h.script.push([det(0, 0)]);
    let r = h.send("a.jpg").wait();
    assert!(r.success && r.predictions.is_empty(), "{r:?}");
    // IR frame: the lower night threshold keeps it, on this model.
    h.script.push([det(0, 0)]);
    let r = h.send_at(gray_jpeg(), "a.jpg", Instant::now()).wait();
    assert_eq!(label(&r), "c0");
    assert_eq!(h.script.submits(), 2);
    assert!(h.stop() < SOON);
}

/// Live settings of a test model: a long timeout and `night_model`.
fn live_with_night(night: Option<&str>) -> LiveSettings {
    LiveSettings {
        request_timeout: Duration::from_secs(30),
        night_model: night.map(str::to_string),
        ..LiveSettings::default()
    }
}

/// A primary model "day" with `night_model: night` and the model "night", both served
/// through one [`Peers`] directory (as the registry publishes it).
fn day_and_night(night_live: LiveSettings) -> (Harness, Harness, Peers) {
    let peers = Peers::new();
    let night = Harness::start_named("night", 1, night_live, peers.clone());
    let day = Harness::start_named("day", 1, live_with_night(Some("Night")), peers.clone());
    peers.publish([&day.handle, &night.handle]);
    (day, night, peers)
}

#[test]
fn ir_frame_is_answered_by_the_night_model_and_day_frame_by_the_primary() {
    let (day, night, _peers) = day_and_night(live_with_night(None));
    night.script.push([det(3, 0)]);
    day.script.push([det(1, 0)]);
    // An IR frame first, a day frame right behind it: neither waits for the other.
    let ir = day.send_at(gray_jpeg(), "ir.jpg", Instant::now());
    let colour = day.send("day.jpg");
    let got = wait_all(vec![ir, colour]);
    assert_eq!(label(&got[0].0), "c3", "answered by the night model");
    assert_eq!(label(&got[1].0), "c1", "answered by the primary");
    // Same response shape: the night model's provider and timings, the module id.
    let r = &got[0].0;
    assert_eq!(r.executionProvider, "OpenVINO CPU (Fake CPU)");
    assert_eq!(r.moduleId, crate::MODULE_ID);
    assert_eq!(r.command, "detect");
    assert_eq!(r.message, "Found 1 objects");
    assert_eq!(night.script.submits(), 1);
    assert_eq!(
        day.script.submits(),
        1,
        "only the day frame ran on the primary"
    );
    assert_eq!(day.handle.metrics.night_routed(), 1);
    assert_eq!(night.handle.metrics.night_routed(), 0);
    assert_eq!(night.handle.metrics.inference_ms.count(), 1);
    for r in RouteFallback::ALL {
        assert_eq!(day.handle.metrics.night_fallback(r), 0, "{r:?}");
    }
    day.stop();
    night.stop();
}

#[test]
fn the_night_models_own_filters_apply_to_forwarded_frames() {
    use crate::model::filters::{FilterRules, SuppressReason};
    let mut live = live_with_night(None);
    live.filters = FilterRules {
        night_confidence_threshold: Some(0.95),
        ..FilterRules::default()
    };
    let (day, night, _peers) = day_and_night(live);
    night.script.push([det(0, 0)]);
    let r = day.send_at(gray_jpeg(), "ir.jpg", Instant::now()).wait();
    // The forwarded stats say IR: the night model's night threshold removed the 0.9 box.
    assert!(r.success && r.predictions.is_empty(), "{r:?}");
    assert_eq!(
        night
            .handle
            .metrics
            .suppressed(SuppressReason::NightThreshold),
        1
    );
    assert_eq!(
        day.handle
            .metrics
            .suppressed(SuppressReason::NightThreshold),
        0
    );
    day.stop();
    night.stop();
}

/// A forwarded frame is the original (not the primary's enhanced copy), preprocessed with
/// the night model's own modes for the scene the primary measured.
#[test]
fn forwarded_frames_use_the_night_models_own_preprocessing() {
    use crate::model::enhance::{Preprocess, PreprocessModes};
    let modes = |night| PreprocessModes {
        preprocess: Preprocess::None,
        night_preprocess: Some(night),
    };
    let mut night_live = live_with_night(None);
    night_live.preprocess = modes(Preprocess::Gamma(0.5));
    let peers = Peers::new();
    let night = Harness::start_named("night", 1, night_live, peers.clone());
    let mut day_live = live_with_night(Some("night"));
    // A strong darkening on the primary: visible in the tensor if it leaked into the frame.
    day_live.preprocess = modes(Preprocess::Gamma(3.0));
    let day = Harness::start_named("day", 1, day_live, peers.clone());
    peers.publish([&day.handle, &night.handle]);
    night.script.push([det(3, 0)]);
    let r = day.send_at(gray_jpeg(), "ir.jpg", Instant::now()).wait();
    assert_eq!(label(&r), "c3");
    let want = |m| {
        // The fake model is a YOLO26: letterbox, no normalization.
        let spec =
            crate::prep::PrepSpec::new(INPUT.0, INPUT.1, crate::model::ResizeMode::Letterbox, None);
        let mut pre = spec.preprocessor();
        let tensors = crate::prep::TensorPool::new();
        let p = crate::prep::PrepPool::prepare_now(
            &mut pre,
            &tensors,
            &gray_jpeg(),
            &spec,
            false,
            modes(m),
        )
        .unwrap();
        p.chw.to_vec()
    };
    let got = night.script.inputs();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0], want(Preprocess::Gamma(0.5)));
    assert_ne!(got[0], want(Preprocess::None), "the enhancement ran");
    assert_eq!(day.script.submits(), 0);
    day.stop();
    night.stop();
}

#[test]
fn unavailable_night_model_falls_back_to_the_primary() {
    use crossbeam_channel::bounded;
    use std::sync::atomic::AtomicUsize;
    let peers = Peers::new();
    let day = Harness::start_named("day", 1, live_with_night(Some("night")), peers.clone());
    let m = || Arc::new(ModelMetrics::new("night", "auto", ""));
    let ir = |h: &Harness| h.send_at(gray_jpeg(), "ir.jpg", Instant::now());
    // Unknown (nothing published yet).
    day.script.push([det(0, 0)]);
    assert_eq!(label(&ir(&day).wait()), "c0");
    assert_eq!(day.handle.metrics.night_fallback(RouteFallback::Unknown), 1);
    // Failed / loading.
    let failed = Arc::new(WorkerHandle::failed("night", "broken", m()));
    peers.publish([&day.handle, &failed]);
    day.script.push([det(1, 0)]);
    assert_eq!(label(&ir(&day).wait()), "c1");
    assert_eq!(
        day.handle.metrics.night_fallback(RouteFallback::NotReady),
        1
    );
    // Ready but its queue is full.
    let (tx, _keep_rx) = bounded(1);
    let mut full = WorkerHandle::failed("night", "unused", m());
    full.state.set(ModelState::Ready);
    full.sender = tx;
    full.soft_capacity = Arc::new(AtomicUsize::new(1));
    let (tx2, _r) = oneshot::channel();
    full.sender
        .try_send(WorkItem::Detect(
            VisionDetectionRequest::default(),
            tx2,
            Instant::now(),
        ))
        .unwrap();
    let full = Arc::new(full);
    peers.publish([&day.handle, &full]);
    day.script.push([det(2, 0)]);
    assert_eq!(label(&ir(&day).wait()), "c2");
    assert_eq!(
        day.handle.metrics.night_fallback(RouteFallback::QueueFull),
        1
    );
    // Ready, room, but its worker is gone (a reload in between).
    let (tx, rx) = bounded(4);
    drop(rx);
    let mut gone = WorkerHandle::failed("night", "unused", m());
    gone.state.set(ModelState::Ready);
    gone.sender = tx;
    gone.soft_capacity = Arc::new(AtomicUsize::new(4));
    let gone = Arc::new(gone);
    peers.publish([&day.handle, &gone]);
    day.script.push([det(3, 0)]);
    assert_eq!(label(&ir(&day).wait()), "c3");
    assert_eq!(day.handle.metrics.night_fallback(RouteFallback::Stopped), 1);
    // A dropped handle (retired worker) is not kept alive by the directory.
    drop(gone);
    day.script.push([det(4, 0)]);
    assert_eq!(label(&ir(&day).wait()), "c4");
    assert_eq!(day.handle.metrics.night_fallback(RouteFallback::Unknown), 2);
    assert_eq!(day.handle.metrics.night_routed(), 0);
    assert_eq!(day.script.submits(), 5);
    drop((failed, full));
    day.stop();
}

#[test]
fn forwarded_requests_are_never_forwarded_again() {
    // A cycle the config validation rejects; the workers must still not bounce frames.
    let (day, night, _peers) = day_and_night(live_with_night(Some("day")));
    night.script.push([det(2, 0)]);
    let r = day.send_at(gray_jpeg(), "ir.jpg", Instant::now()).wait();
    assert_eq!(label(&r), "c2");
    assert_eq!(day.script.submits(), 0);
    assert_eq!(night.script.submits(), 1);
    assert_eq!(day.handle.metrics.night_routed(), 1);
    assert_eq!(night.handle.metrics.night_routed(), 0);
    for r in RouteFallback::ALL {
        assert_eq!(night.handle.metrics.night_fallback(r), 0, "{r:?}");
    }
    // Pointing at itself is ignored too.
    let peers = Peers::new();
    let me = Harness::start_named("me", 1, live_with_night(Some("ME")), peers.clone());
    peers.publish([&me.handle]);
    me.script.push([det(1, 0)]);
    assert_eq!(
        label(&me.send_at(gray_jpeg(), "ir.jpg", Instant::now()).wait()),
        "c1"
    );
    assert_eq!(me.handle.metrics.night_fallback(RouteFallback::Unknown), 1);
    me.stop();
    day.stop();
    night.stop();
}

/// The cascade API: another worker sends a decoded frame and gets the answer on a crossbeam
/// channel; a spent deadline is a drop.
#[test]
fn decoded_frames_can_be_answered_to_another_worker() {
    let h = Harness::start(1, Duration::from_secs(30));
    let img = crate::image::decode(&gray_jpeg()).unwrap();
    let stats = crate::image_stats::ImageStats::compute(&img);
    assert_ne!(stats.scene(), crate::image_stats::Scene::Day);
    let decoded = |deadline: Instant, reply: super::Reply| {
        WorkItem::DetectDecoded(Box::new(Decoded {
            img: img.clone(),
            stats,
            image_name: "crop.jpg".into(),
            min_confidence: 0.0,
            request_id: "abcd1234".into(),
            enqueued: Instant::now(),
            deadline,
            decode: Duration::from_millis(3),
            reply,
            from: "primary".into(),
            purpose: super::Purpose::Verify,
            original: None,
        }))
    };
    h.script.push([det(4, 0)]);
    let (tx, rx) = crossbeam_channel::bounded(1);
    h.handle
        .sender
        .try_send(decoded(Instant::now() + SOON, super::Reply::Worker(tx)))
        .unwrap();
    let r = rx.recv_timeout(SOON).unwrap();
    assert_eq!(label(&r), "c4");
    // Letterboxed for this model like a decoded upload: same box as from the JPEG.
    assert_eq!(
        (r.predictions[0].x_min, r.predictions[0].y_min),
        (10, 10),
        "{r:?}"
    );
    let (tx, rx) = crossbeam_channel::bounded(1);
    h.handle
        .sender
        .try_send(decoded(Instant::now(), super::Reply::Worker(tx)))
        .unwrap();
    let r = rx.recv_timeout(SOON).unwrap();
    assert!(!r.success);
    assert!(
        r.message.starts_with("Request dropped: waited"),
        "{}",
        r.message
    );
    assert_eq!(h.script.submits(), 1);
    h.stop();
}

// ---- Verifier cascade (`verify`) ----

use crate::verify::{Outcome, VerifyConfig, VerifyMode};

/// A box covering the whole model input: in a crop it is the whole crop, which overlaps the
/// centred candidate box with IoU 1 / (1 + 2 * 0.2)^2 = 0.51 (pad 0.2).
fn whole(class: usize, score: f32, delay_ms: u64) -> Step {
    Step::detections(
        &[[0.0, 0.0, 64.0, 64.0, score, class as f32]],
        Duration::from_millis(delay_ms),
    )
}

fn nothing() -> Step {
    Step::detections(&[], Duration::ZERO)
}

/// Live settings of a primary verified by `verifier` (labels `c1`, the other defaults).
fn live_verified(verifier: &str, tweak: impl FnOnce(&mut VerifyConfig)) -> LiveSettings {
    let mut v = VerifyConfig {
        labels: vec!["C1".into()],
        ..VerifyConfig::with_model(verifier)
    };
    tweak(&mut v);
    LiveSettings {
        request_timeout: Duration::from_secs(30),
        verify: Some(v),
        ..LiveSettings::default()
    }
}

/// "primary" verified by "checker" (`checker_live`), both in one [`Peers`] directory.
fn primary_and_checker(
    primary_live: LiveSettings,
    checker_live: LiveSettings,
) -> (Harness, Harness, Peers) {
    let peers = Peers::new();
    let checker = Harness::start_named("checker", 1, checker_live, peers.clone());
    let primary = Harness::start_named("primary", 1, primary_live, peers.clone());
    peers.publish([&primary.handle, &checker.handle]);
    (primary, checker, peers)
}

fn ir(h: &Harness, name: &str) -> Reply {
    h.send_at(gray_jpeg(), name, Instant::now())
}

fn labels(r: &VisionDetectionResponse) -> Vec<&str> {
    assert!(r.success, "{}", r.message);
    assert_eq!(r.count as usize, r.predictions.len());
    assert_eq!(r.message, format!("Found {} objects", r.predictions.len()));
    r.predictions.iter().map(|p| p.label.as_str()).collect()
}

#[test]
fn verifier_keeps_confirmed_and_drops_rejected_candidates() {
    use crate::model::filters::{FilterRules, SuppressReason};
    // The checker's own rules must not apply to checks: a night threshold above every score,
    // an object filter without c1, and a verifier of its own pointing back at the primary.
    let mut checker_live = live_verified("primary", |_| {});
    checker_live.filters = FilterRules {
        night_confidence_threshold: Some(0.99),
        ..FilterRules::default()
    };
    checker_live.object_filter = vec!["c4".into()];
    let (primary, checker, _peers) =
        primary_and_checker(live_verified("checker", |_| {}), checker_live);
    // Confirmed; answered with nothing; confirmed with another label; confirmed too unsure.
    checker.script.push([
        whole(1, 0.9, 0),
        nothing(),
        whole(2, 0.9, 0),
        whole(1, 0.3, 0),
    ]);
    primary.script.push((0..4).map(|_| det(1, 0)));
    let got: Vec<Vec<String>> = (0..4)
        .map(|i| {
            labels(&ir(&primary, &format!("ir{i}.jpg")).wait())
                .into_iter()
                .map(str::to_string)
                .collect()
        })
        .collect();
    assert_eq!(got, [vec!["c1".to_string()], vec![], vec![], vec![]]);
    // A non-candidate passes unchanged next to a rejected candidate.
    checker.script.push([nothing()]);
    primary.script.push([Step::detections(
        &[
            [10.0, 18.0, 30.0, 38.0, 0.9, 0.0],
            [40.0, 18.0, 60.0, 38.0, 0.8, 1.0],
        ],
        Duration::ZERO,
    )]);
    assert_eq!(labels(&ir(&primary, "two.jpg").wait()), ["c0"]);
    let m = &primary.handle.metrics;
    assert_eq!(m.verified(Outcome::Kept), 1);
    assert_eq!(m.verified(Outcome::Rejected), 4);
    assert_eq!(m.verified(Outcome::Unverified), 0);
    assert_eq!(m.suppressed(SuppressReason::Verify), 4);
    assert_eq!(checker.script.submits(), 5, "one crop per candidate");
    assert_eq!(
        primary.script.submits(),
        5,
        "checks are never verified again"
    );
    // The checker saw the padded crop: 20x20 box + 4 px on each side.
    let inputs = checker.script.inputs();
    assert_eq!(inputs.len(), 5);
    let frame = crate::image::decode(&gray_jpeg()).unwrap();
    let stats = crate::image_stats::ImageStats::compute(&frame);
    let rect = crate::verify::CropRect {
        x: 6,
        y: 6,
        w: 28,
        h: 28,
    };
    assert_eq!(
        crate::verify::crop_rect([10.0, 10.0, 30.0, 30.0], 0.2, 64, 48),
        rect
    );
    let spec =
        crate::prep::PrepSpec::new(INPUT.0, INPUT.1, crate::model::ResizeMode::Letterbox, None);
    let mut pre = spec.preprocessor();
    let want = crate::prep::PrepPool::prepare_image(
        &mut pre,
        &crate::prep::TensorPool::new(),
        crate::verify::crop(&frame, rect),
        stats,
        &spec,
        false,
        Default::default(),
    )
    .unwrap();
    assert_eq!(inputs[0], want.chw.to_vec());
    assert_eq!(
        checker
            .handle
            .metrics
            .suppressed(SuppressReason::NightThreshold),
        0
    );
    assert_eq!(checker.handle.metrics.verified(Outcome::Kept), 0);
    primary.stop();
    checker.stop();
}

#[test]
fn day_frames_are_not_verified_when_night_only() {
    let (primary, checker, _peers) =
        primary_and_checker(live_verified("checker", |_| {}), live_with_night(None));
    primary.script.push([det(1, 0)]);
    assert_eq!(labels(&primary.send("day.jpg").wait()), ["c1"]);
    assert_eq!(checker.script.submits(), 0);
    assert_eq!(primary.handle.metrics.verified(Outcome::Kept), 0);
    // With night_only off the day frame is checked too.
    assert!(
        primary
            .handle
            .set_live(live_verified("checker", |v| v.night_only = false))
    );
    checker.script.push([nothing()]);
    primary.script.push([det(1, 0)]);
    assert!(labels(&primary.send("day.jpg").wait()).is_empty());
    assert_eq!(checker.script.submits(), 1);
    primary.stop();
    checker.stop();
}

#[test]
fn unavailable_verifier_leaves_candidates_unverified() {
    let peers = Peers::new();
    let primary = Harness::start_named(
        "primary",
        1,
        live_verified("checker", |_| {}),
        peers.clone(),
    );
    // Not served.
    primary.script.push([det(1, 0)]);
    assert_eq!(labels(&ir(&primary, "a.jpg").wait()), ["c1"]);
    // Failed (not ready).
    let failed = Arc::new(WorkerHandle::failed(
        "checker",
        "broken",
        Arc::new(ModelMetrics::new("checker", "auto", "")),
    ));
    peers.publish([&primary.handle, &failed]);
    primary.script.push([det(1, 0)]);
    assert_eq!(labels(&ir(&primary, "b.jpg").wait()), ["c1"]);
    // The model itself is never its verifier.
    assert!(primary.handle.set_live(live_verified("PRIMARY", |_| {})));
    primary.script.push([det(1, 0)]);
    assert_eq!(labels(&ir(&primary, "c.jpg").wait()), ["c1"]);
    let m = &primary.handle.metrics;
    assert_eq!(m.verified(Outcome::Unverified), 3);
    assert_eq!(m.verified(Outcome::Rejected), 0);
    drop(failed);
    primary.stop();
}

#[test]
fn verifier_past_the_deadline_leaves_candidates_unverified() {
    let (primary, checker, _peers) =
        primary_and_checker(live_verified("checker", |_| {}), live_with_night(None));
    // The request arrived 29.5 s ago (timeout 30 s): 0.5 s are left, the checker needs 3 s.
    checker.script.push([whole(1, 0.9, 3000)]);
    primary.script.push([det(1, 0)]);
    let enqueued = Instant::now()
        .checked_sub(Duration::from_millis(29_500))
        .unwrap_or_else(Instant::now);
    let t = Instant::now();
    let r = primary.send_at(gray_jpeg(), "late.jpg", enqueued).wait();
    let waited = t.elapsed();
    assert_eq!(labels(&r), ["c1"], "kept unverified");
    assert!(waited >= Duration::from_millis(400), "{waited:?}");
    assert!(waited < Duration::from_millis(2500), "{waited:?}");
    assert_eq!(primary.handle.metrics.verified(Outcome::Unverified), 1);
    assert!(r.processMs >= 400, "verify time counts: {}", r.processMs);
    primary.stop();
    checker.stop();
}

#[test]
fn slow_verifier_does_not_delay_other_requests() {
    let (primary, checker, _peers) =
        primary_and_checker(live_verified("checker", |_| {}), live_with_night(None));
    checker.script.push([whole(1, 0.9, 800)]);
    // A has a candidate (slow check), B and C have none, D is a day frame.
    primary
        .script
        .push([det(1, 0), det(0, 0), det(2, 0), det(1, 0)]);
    let t = Instant::now();
    let a = ir(&primary, "a.jpg");
    let b = ir(&primary, "b.jpg");
    let c = ir(&primary, "c.jpg");
    let d = primary.send("d.jpg");
    let got = wait_all(vec![a, b, c, d]);
    assert_eq!(labels(&got[0].0), ["c1"], "confirmed");
    assert_eq!(labels(&got[1].0), ["c0"]);
    assert_eq!(labels(&got[2].0), ["c2"]);
    assert_eq!(labels(&got[3].0), ["c1"], "day frame, not verified");
    // B, C and D were answered while A's check was still running.
    for (i, (_, at)) in got.iter().enumerate().skip(1) {
        assert!(*at < got[0].1, "request {i} waited for the verification");
    }
    assert!(got[0].1 - t >= Duration::from_millis(750));
    assert!(got[0].0.processMs >= 750, "{}", got[0].0.processMs);
    primary.stop();
    checker.stop();
}

#[test]
fn frame_mode_matches_candidates_by_iou_on_the_whole_frame() {
    let (primary, checker, _peers) = primary_and_checker(
        live_verified("checker", |v| v.mode = VerifyMode::Frame),
        live_with_night(None),
    );
    // The same place as the primary's c1 box (frame [10, 10, 30, 30]); then elsewhere.
    checker.script.push([
        det(1, 0),
        Step::detections(&[[40.0, 18.0, 60.0, 38.0, 0.9, 1.0]], Duration::ZERO),
    ]);
    primary.script.push([det(1, 0), det(1, 0)]);
    assert_eq!(labels(&ir(&primary, "a.jpg").wait()), ["c1"]);
    assert!(labels(&ir(&primary, "b.jpg").wait()).is_empty());
    // The whole frame went to the checker, letterboxed like an upload.
    let want = {
        let spec =
            crate::prep::PrepSpec::new(INPUT.0, INPUT.1, crate::model::ResizeMode::Letterbox, None);
        let mut pre = spec.preprocessor();
        let tensors = crate::prep::TensorPool::new();
        crate::prep::PrepPool::prepare_now(
            &mut pre,
            &tensors,
            &gray_jpeg(),
            &spec,
            false,
            Default::default(),
        )
        .unwrap()
        .chw
        .to_vec()
    };
    assert_eq!(checker.script.inputs()[0], want);
    primary.stop();
    checker.stop();
}

#[test]
fn routed_night_frames_use_the_night_models_verifier() {
    let peers = Peers::new();
    let checker = Harness::start_named("checker", 1, live_with_night(None), peers.clone());
    let night = Harness::start_named("night", 1, live_verified("checker", |_| {}), peers.clone());
    // The primary's own verify is not used for frames the night model answers.
    let mut day_live = live_verified("nobody", |_| {});
    day_live.night_model = Some("night".into());
    let day = Harness::start_named("day", 1, day_live, peers.clone());
    peers.publish([&day.handle, &night.handle, &checker.handle]);
    night.script.push([det(1, 0)]);
    checker.script.push([nothing()]);
    assert!(labels(&ir(&day, "ir.jpg").wait()).is_empty());
    assert_eq!(night.handle.metrics.verified(Outcome::Rejected), 1);
    assert_eq!(day.handle.metrics.verified(Outcome::Unverified), 0);
    assert_eq!(checker.script.submits(), 1);
    day.stop();
    night.stop();
    checker.stop();
}

/// Dataset collector end to end on the fake runtime: requests produce sidecars with the right
/// buckets (positive, suppressed, empty, near-threshold below the threshold, verifier
/// rejection), and a collector that cannot keep up (paused, its queue full) never delays a
/// reply. The only test that drives the process-wide collector.
#[test]
fn collected_frames_get_their_buckets_and_a_full_queue_never_delays_replies() {
    use crate::collect::metrics::{Dropped, METRICS};
    use crate::collect::{Bucket, CollectConfig, CollectSettings, Record};
    use crate::model::filters::FilterRules;
    let dir = std::env::temp_dir().join(format!("bop-collect-e2e-{}", uuid::Uuid::new_v4()));
    let collect = Arc::new(CollectSettings::from_config(
        &CollectConfig {
            enabled: true,
            dir: Some(dir.clone()),
            ..CollectConfig::default()
        },
        &dir,
    ));
    let mut live = live_verified("checker", |_| {});
    live.collect = collect.clone();
    live.filters = FilterRules {
        label_thresholds: [("c2".to_string(), 0.95)].into_iter().collect(),
        ..FilterRules::default()
    };
    let checker_live = LiveSettings {
        request_timeout: Duration::from_secs(30),
        ..LiveSettings::default()
    };
    let (primary, checker, _peers) = primary_and_checker(live, checker_live);

    // 1. The collector is stuck: every reply still arrives at once, the overflow is counted.
    crate::collect::PAUSE.store(true, std::sync::atomic::Ordering::Relaxed);
    let full_before = METRICS.dropped(Dropped::QueueFull);
    let n = crate::collect::QUEUE + 12;
    primary.script.push((0..n).map(|_| nothing()));
    let t = Instant::now();
    let replies: Vec<Reply> = (0..n)
        .map(|i| primary.send(&format!("flood{i}.jpg")))
        .collect();
    for (r, _) in wait_all(replies) {
        assert!(r.success, "{}", r.message);
    }
    assert!(t.elapsed() < SOON);
    let dropped = METRICS.dropped(Dropped::QueueFull) - full_before;
    assert!(
        dropped >= (n - crate::collect::QUEUE - 1) as u64,
        "dropped {dropped}"
    );
    crate::collect::PAUSE.store(false, std::sync::atomic::Ordering::Relaxed);
    // The flood frames share the camera `flood` (trailing digits are not part of the camera):
    // one is saved (`empty`), the others are rate limited. Wait until the queue drained.
    primary.wait_for("the collector to drain", || {
        crate::collect::global().pending() == 0
    });

    // 2. One request per camera and outcome.
    primary.script.push([det(0, 0)]); // positive
    assert_eq!(label(&primary.send("front.jpg").wait()), "c0");
    primary.script.push([det(2, 0)]); // c2 under its label threshold: suppressed
    assert!(primary.send("side.jpg").wait().predictions.is_empty());
    primary.script.push([nothing()]); // empty
    assert!(primary.send("back.jpg").wait().predictions.is_empty());
    primary.script.push([whole(0, 0.45, 0)]); // below 0.5, within 0.1: empty + near
    assert!(primary.send("yard.jpg").wait().predictions.is_empty());
    // An IR c1 candidate the checker does not confirm: rejected by the verifier.
    primary.script.push([whole(1, 0.9, 0)]);
    checker.script.push([nothing()]);
    assert!(ir(&primary, "gate.jpg").wait().predictions.is_empty());

    let sidecar = |scene: &str, camera: &str| -> Record {
        let t = Instant::now();
        loop {
            let base = dir.join(scene).join(camera);
            let found = std::fs::read_dir(&base).ok().and_then(|days| {
                days.flatten().find_map(|d| {
                    std::fs::read_dir(d.path()).ok()?.flatten().find_map(|f| {
                        let p = f.path();
                        (p.extension()? == "json")
                            .then(|| std::fs::read(&p).ok())
                            .flatten()
                    })
                })
            });
            if let Some(bytes) = found {
                return serde_json::from_slice(&bytes).unwrap();
            }
            assert!(
                t.elapsed() < SOON,
                "no sidecar for {scene}/{camera}; drops {:?}; files {:?}",
                Dropped::ALL.map(|d| (d.as_str(), METRICS.dropped(d))),
                crate::collect::layout::day_dirs(&dir)
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    let r = sidecar("day", "front");
    assert_eq!(r.buckets, vec![Bucket::Positive]);
    assert_eq!(r.model, "primary");
    assert_eq!(r.raw.len(), 1);
    assert_eq!(r.predictions[0].label, "c0");
    assert_eq!(r.camera_key, "primary|cam:front");
    assert!(r.image.original);
    let r = sidecar("day", "side");
    assert_eq!(r.buckets, vec![Bucket::Suppressed]);
    assert_eq!(r.suppressed[0].reason, "label_threshold");
    assert_eq!(r.suppressed[0].label, "c2");
    let r = sidecar("day", "back");
    assert_eq!(r.buckets, vec![Bucket::Empty]);
    let r = sidecar("day", "yard");
    assert_eq!(r.buckets, vec![Bucket::Empty, Bucket::NearThreshold]);
    assert!(r.raw.is_empty());
    assert_eq!(r.below_threshold.len(), 1);
    assert!((r.below_threshold[0].score - 0.45).abs() < 1e-4);
    let r = sidecar("ir", "gate");
    assert_eq!(
        r.buckets,
        vec![Bucket::VerifyRejected, Bucket::Disagreement]
    );
    let v = r.verify.as_ref().unwrap();
    assert_eq!(v.verifier, "checker");
    assert_eq!(v.candidates[0].outcome, "rejected");
    assert!(r.suppressed.iter().any(|s| s.reason == "verify"));
    let r = sidecar("day", "flood");
    assert_eq!(r.buckets, vec![Bucket::Empty]);
    let summary = crate::collect::Summary::scan(&dir);
    assert_eq!(summary.by_camera["flood"].frames, 1);
    assert_eq!(summary.frames, 6);
    primary.stop();
    checker.stop();
    let _ = std::fs::remove_dir_all(&dir);
}
