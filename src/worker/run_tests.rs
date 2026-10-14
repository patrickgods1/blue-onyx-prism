//! In-process tests of the run loop (PL-12): a real worker thread, spawned like the registry
//! does (`spawn_worker`, bounded crossbeam channel, shared `PrepPool`), on the scripted
//! `backend::fake` runtime. They cover FIFO replies with several slots, a prep failure in the
//! middle, the timeout before submit, shutdown with a busy slot, and the stuck-inference path.

use super::watch::STUCK_PREFIX;
use super::*;
use crate::backend::fake::{Call, FakeScript, Step, yolo26_output};
use crate::image::RgbImage;
use crate::model::ModelFamilyKind;
use tokio::sync::oneshot;

/// Model input (width, height) of the fake model.
const INPUT: (u32, u32) = (64, 64);
/// Generous bound for anything that should happen "at once" (CI runners are slow).
const SOON: Duration = Duration::from_secs(10);

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
    handle: WorkerHandle,
    script: FakeScript,
    shutdown: CancellationToken,
}

impl Harness {
    /// Spawn a worker on the fake runtime and wait until its model is ready.
    fn start(slots: usize, request_timeout: Duration) -> Self {
        let script = FakeScript::new(slots, INPUT);
        let runtimes = Arc::new(Mutex::new(Runtimes::fake(script.clone())));
        let path = PathBuf::from("fake.onnx");
        let cfg = WorkerConfig {
            name: "fake".into(),
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
            live: LiveSettings {
                request_timeout,
                ..LiveSettings::default()
            }
            .shared(),
            queue_size: 16,
            lazy: false,
            can_use_gpu: false,
            wait: None,
            concurrency: None,
            prep: Some(PrepPool::new(2)),
        };
        let metrics = Arc::new(ModelMetrics::new("fake", "auto", ""));
        let shutdown = CancellationToken::new();
        let handle = spawn_worker(runtimes, cfg, metrics, shutdown.clone());
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
        self.handle.join();
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
