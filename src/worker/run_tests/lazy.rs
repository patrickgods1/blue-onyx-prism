//! Run-loop tests of the paths before a model's first load (QA-150, QA-273): a lazy model
//! (served at once, compiled on `Load`, its handle's gate forwarding the startup order) and a
//! model that waits for its downloads (a scripted [`Wait`]: gate opened first, progress in the
//! state, failure, a runtime awaiting restart, classes read after the wait, shutdown).

use super::*;
use crate::metrics::DropReason;
use crate::resources::manager::{Manager, ManagerOptions};
use crate::resources::provision::{Wait, WaitScript, WaitStatus};
use std::sync::atomic::AtomicUsize;

/// The fake model's worker config (as `spawn_on`), eager and loading at once.
fn worker_cfg(faults: TestFaults) -> WorkerConfig {
    let path = PathBuf::from("fake.onnx");
    WorkerConfig {
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
            no_cache: false,
        },
        classes: classes(),
        live: LiveSettings::default().shared(),
        queue_size: QUEUE,
        lazy: false,
        can_use_gpu: false,
        wait: None,
        concurrency: None,
        prep: Some(PrepPool::new(1)),
        peers: Peers::new(),
        only: None,
        make_room: None,
        recovering_from: None,
        faults,
    }
}

/// Spawn `cfg` on `script`'s fake runtime after `after`.
fn spawn_cfg(
    script: &FakeScript,
    cfg: WorkerConfig,
    after: Option<LoadGate>,
) -> (WorkerHandle, CancellationToken) {
    let runtimes = Arc::new(Mutex::new(Runtimes::fake(script.clone())));
    let shutdown = CancellationToken::new();
    let metrics = Arc::new(ModelMetrics::new(&cfg.name, "auto", ""));
    let handle = spawn_worker_after(runtimes, cfg, metrics, shutdown.clone(), after);
    (handle, shutdown)
}

/// Send a detection request for `name` to `handle`.
fn send_to(handle: &WorkerHandle, name: &str) -> Reply {
    let (tx, rx) = oneshot::channel();
    let req = VisionDetectionRequest {
        min_confidence: 0.0,
        image_data: jpeg().into(),
        image_name: name.into(),
        ..VisionDetectionRequest::default()
    };
    handle
        .sender
        .try_send(WorkItem::Detect(req, tx, Instant::now()))
        .unwrap();
    Reply(rx)
}

/// A lazy model is served before it loads (requests answered "initializing" at once), compiles
/// on `Load` only and then runs on every slot of its device; its handle's gate is the earlier
/// model's, so the next model in the config waits for that one, not for this lazy one.
#[test]
fn a_lazy_model_compiles_on_load_and_uses_every_slot() {
    let script = FakeScript::new(2, INPUT);
    let earlier = LoadGate::new();
    let cfg = WorkerConfig {
        lazy: true,
        ..worker_cfg(TestFaults::default())
    };
    let (handle, shutdown) = spawn_cfg(&script, cfg, Some(earlier.clone()));
    assert!(handle.accepts_while_initializing());
    assert!(!handle.load_gate().is_open(), "the earlier model's gate");

    // Served at once, although the earlier model has not loaded.
    let r = send_to(&handle, "early.jpg").wait();
    assert!(!r.success);
    assert_eq!(r.dropped, Some(DropReason::Initializing));
    assert_eq!(r.message, "Model 'fake' is initializing");
    assert_eq!(handle.state.get(), ModelState::Initializing);

    earlier.open();
    assert!(handle.load_gate().is_open());
    assert!(script.compiles().is_empty(), "nothing compiles before Load");

    assert!(handle.trigger_load());
    wait_until("the lazy load", || handle.state.get() == ModelState::Ready);
    assert_eq!(script.compiles().len(), 1);

    // Both slots run: the second request is submitted while the first is in flight.
    script.push([det(0, 150), det(1, 0)]);
    let got = wait_all(vec![send_to(&handle, "a.jpg"), send_to(&handle, "b.jpg")]);
    assert_eq!(label(&got[0].0), "c0");
    assert_eq!(label(&got[1].0), "c1");
    let slots: Vec<usize> = script
        .calls()
        .iter()
        .filter_map(|c| match c {
            Call::Submit { slot, .. } => Some(*slot),
            _ => None,
        })
        .collect();
    assert_eq!(slots, [0, 1], "{:?}", script.calls());
    shutdown.cancel();
    joined_within_soon(handle);
}

/// A lazy model whose load failed is not compiled again by a second `Load` (the server's
/// trigger fires once; a `Load` straight into the channel stands for a later one).
#[test]
fn a_failed_lazy_load_is_not_tried_again_by_another_load() {
    let script = FakeScript::new(1, INPUT);
    let tries = Arc::new(AtomicUsize::new(0));
    let faults = TestFaults {
        load: Some(Arc::new({
            let tries = tries.clone();
            move |_: &Candidate| {
                tries.fetch_add(1, Ordering::SeqCst);
                anyhow::bail!("no device")
            }
        })),
        ..TestFaults::default()
    };
    let cfg = WorkerConfig {
        lazy: true,
        ..worker_cfg(faults)
    };
    let (handle, shutdown) = spawn_cfg(&script, cfg, None);
    assert!(handle.trigger_load());
    wait_until("the failed load", || {
        matches!(handle.state.get(), ModelState::Failed(_))
    });
    assert_eq!(tries.load(Ordering::SeqCst), 1);
    let failed = handle.state.get();

    handle.sender.try_send(WorkItem::Load).unwrap();
    // Taken in order: once this is answered, the Load before it was handled.
    let r = send_to(&handle, "after.jpg").wait();
    assert_eq!(r.dropped, Some(DropReason::Failed));
    assert_eq!(tries.load(Ordering::SeqCst), 1, "compiled again");
    assert_eq!(handle.state.get(), failed);
    assert!(script.compiles().is_empty());
    shutdown.cancel();
    joined_within_soon(handle);
}

/// A model waiting for downloads on `script`, with `classes` (a path; None: none configured)
/// and the config's class list empty, as the planner leaves it.
fn spawn_waiting(
    fake: &FakeScript,
    script: WaitScript,
    classes: Option<PathBuf>,
    downloads: &std::path::Path,
) -> (WorkerHandle, CancellationToken) {
    let manager = Arc::new(Manager::start(ManagerOptions::new(downloads)));
    let mut cfg = worker_cfg(TestFaults::default());
    cfg.classes = Vec::new();
    cfg.model.classes = classes;
    cfg.wait = Some(Wait::scripted(manager, Vec::new(), script));
    spawn_cfg(fake, cfg, None)
}

/// The progress goes into the state as a download wait, the next model is not held up while
/// it downloads (gate open first), and the model loads once the downloads are in, with the
/// class names read from disk then.
#[test]
fn a_waiting_model_shows_progress_then_loads_with_its_classes() {
    let dir = crate::test_dirs::TempDir::new("wait");
    let names = dir.join("names.yaml");
    std::fs::write(&names, "names: [alpha, beta]\n").unwrap();
    let fake = FakeScript::new(1, INPUT);
    let script = WaitScript::new([WaitStatus::Pending("x 10%".into())]);
    let (handle, shutdown) =
        spawn_waiting(&fake, script.clone(), Some(names), &dir.join("downloads"));
    wait_until("the progress", || {
        handle.state.detail_kind() == Some(("x 10%".into(), true))
    });
    assert!(handle.load_gate().is_open(), "the next model is held up");
    assert!(fake.compiles().is_empty());

    script.push(WaitStatus::Ready);
    wait_until("the load", || handle.state.get() == ModelState::Ready);
    assert_eq!(fake.compiles().len(), 1);
    fake.push([det(1, 0)]);
    assert_eq!(label(&send_to(&handle, "a.jpg").wait()), "beta");
    shutdown.cancel();
    joined_within_soon(handle);
}

/// A download that failed for good makes the model Failed with the manager's message.
#[test]
fn a_failed_download_fails_the_model() {
    let dir = crate::test_dirs::TempDir::new("wait");
    let fake = FakeScript::new(1, INPUT);
    let script = WaitScript::new([WaitStatus::Failed("boom".into())]);
    let (handle, shutdown) = spawn_waiting(&fake, script, None, &dir.join("downloads"));
    wait_until("the failure", || {
        matches!(handle.state.get(), ModelState::Failed(_))
    });
    assert_eq!(handle.state.get(), ModelState::Failed("boom".into()));
    assert!(handle.load_gate().is_open());
    assert!(fake.compiles().is_empty());
    shutdown.cancel();
    joined_within_soon(handle);
}

/// A classes file that cannot be read once the files are in fails the model with the fix.
#[test]
fn a_bad_classes_file_after_the_download_fails_the_model() {
    let dir = crate::test_dirs::TempDir::new("wait");
    let fake = FakeScript::new(1, INPUT);
    let script = WaitScript::new([WaitStatus::Ready]);
    let missing = dir.join("missing.yaml");
    let (handle, shutdown) = spawn_waiting(&fake, script, Some(missing), &dir.join("downloads"));
    wait_until("the failure", || {
        matches!(handle.state.get(), ModelState::Failed(_))
    });
    let ModelState::Failed(msg) = handle.state.get() else {
        unreachable!();
    };
    assert!(msg.contains("the classes file is missing"), "{msg}");
    assert!(msg.contains("missing.yaml"), "{msg}");
    assert!(fake.compiles().is_empty());
    shutdown.cancel();
    joined_within_soon(handle);
}

/// A runtime it needs got installed: the model stays initializing with the restart note and
/// never loads in this generation; a stop ends the wait.
#[test]
fn a_model_awaiting_a_restart_never_loads_and_stops_while_waiting() {
    let dir = crate::test_dirs::TempDir::new("wait");
    let fake = FakeScript::new(1, INPUT);
    let note = "ONNX Runtime installed; restarting to load it";
    let script = WaitScript::new([WaitStatus::AwaitingRestart(note.into())]);
    let (handle, shutdown) = spawn_waiting(&fake, script.clone(), None, &dir.join("downloads"));
    wait_until("a few polls", || script.polls() >= 3);
    assert_eq!(handle.state.get(), ModelState::Initializing);
    assert_eq!(handle.state.detail_kind(), Some((note.into(), true)));
    assert!(handle.load_gate().is_open());
    assert!(fake.compiles().is_empty());
    let polls = script.polls();
    shutdown.cancel();
    joined_within_soon(handle);
    assert!(script.polls() <= polls + 1, "polled on after the stop");
    assert!(fake.compiles().is_empty());
}
