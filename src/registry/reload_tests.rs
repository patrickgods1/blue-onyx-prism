//! Reloads on the scripted fake runtime: real worker threads and a real registry, with device
//! candidates and load failures injected ([`crate::worker::TestFaults`]). What is checked is
//! order (who serves when, what is retired first), never speed.

use super::*;
use crate::api::{VisionDetectionRequest, VisionDetectionResponse};
use crate::backend::fake::{FakeScript, Step};
use crate::backend::plan::LoadFailure;
use crate::backend::spec::{Device, Runtime, Target};
use crate::image::RgbImage;
use crate::model::ModelFamilyKind;
use crate::supervisor::Status;
use crate::worker::{TestFaults, WorkItem};
use std::collections::VecDeque;
use std::sync::Condvar;

const SOON: Duration = Duration::from_secs(10);
const OOM: &str = "[GPU] clEnqueueMapBuffer, error code: -5 CL_OUT_OF_RESOURCES";
const LOST: &str = "DXGI_ERROR_DEVICE_REMOVED";
const NO_OP: &str = "[GPU] Unsupported operation: Foo";

fn gpu() -> Device {
    Device::new(Runtime::OpenVino, Target::Gpu, None).unwrap()
}

fn cpu() -> Device {
    Device::OPENVINO_CPU
}

fn wait_until(what: &str, cond: impl Fn() -> bool) {
    let t = Instant::now();
    while !cond() {
        assert!(t.elapsed() < SOON * 3, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// What each compile does: the scripted load faults of a registry.
#[derive(Default)]
struct LoadState {
    /// Every try, in order: its device.
    tried: Vec<String>,
    /// Errors the next tries of a device fail with.
    fail: HashMap<String, VecDeque<String>>,
    /// Tries block while the count of tries is at least this.
    hold_from: Option<usize>,
    /// Runs at each try, with that try's number: to look at the world from inside a load.
    probe: Option<Arc<dyn Fn(usize) + Send + Sync>>,
}

#[derive(Clone, Default)]
struct Loads(Arc<(Mutex<LoadState>, Condvar)>);

impl Loads {
    fn at(&self, cand: &Candidate) -> Result<()> {
        let device = cand.device.to_string();
        let (lock, cv) = &*self.0;
        let mut st = lock.lock_recover();
        st.tried.push(device.clone());
        let this_try = st.tried.len();
        cv.notify_all();
        let probe = st.probe.clone();
        drop(st);
        if let Some(p) = probe {
            p(this_try);
        }
        let deadline = Instant::now() + SOON * 3;
        let mut st = lock.lock_recover();
        while st.hold_from.is_some_and(|n| this_try > n) && Instant::now() < deadline {
            st = cv
                .wait_timeout(st, Duration::from_millis(20))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        match st.fail.get_mut(&device).and_then(VecDeque::pop_front) {
            Some(e) => Err(anyhow::anyhow!(e)),
            None => Ok(()),
        }
    }

    fn fail(&self, device: Device, errors: &[&str]) {
        let mut st = self.0.0.lock_recover();
        st.fail
            .entry(device.to_string())
            .or_default()
            .extend(errors.iter().map(|e| e.to_string()));
    }

    /// Tries after the `n`th block until [`Self::release`].
    fn hold_after(&self, n: usize) {
        self.0.0.lock_recover().hold_from = Some(n);
    }

    fn release(&self) {
        self.0.0.lock_recover().hold_from = None;
        self.0.1.notify_all();
    }

    fn tried(&self) -> Vec<String> {
        self.0.0.lock_recover().tried.clone()
    }

    fn set_probe(&self, probe: impl Fn(usize) + Send + Sync + 'static) {
        self.0.0.lock_recover().probe = Some(Arc::new(probe));
    }
}

/// A 64x48 JPEG.
fn jpeg() -> Vec<u8> {
    let (w, h) = (64u32, 48u32);
    let img = RgbImage {
        width: w,
        height: h,
        rgb: (0..w * h * 3).map(|i| (i % 251) as u8).collect(),
    };
    crate::image::encode_jpeg(&img, 90).unwrap()
}

fn detect(w: &WorkerHandle) -> VisionDetectionResponse {
    let (tx, mut rx) = tokio::sync::oneshot::channel();
    let req = VisionDetectionRequest {
        image_data: jpeg().into(),
        image_name: "x.jpg".into(),
        ..VisionDetectionRequest::default()
    };
    w.sender
        .try_send(WorkItem::Detect(req, tx, Instant::now()))
        .unwrap();
    let t = Instant::now();
    loop {
        match rx.try_recv() {
            Ok(r) => return r,
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                assert!(t.elapsed() < SOON, "no reply");
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(e) => panic!("the worker dropped the reply: {e}"),
        }
    }
}

/// True once the worker thread behind `w` has ended (it holds the only receiver).
fn thread_gone(w: &WorkerHandle) -> bool {
    matches!(
        w.sender.try_send(WorkItem::Load),
        Err(crossbeam_channel::TrySendError::Disconnected(_))
    )
}

struct Fixture {
    reg: Arc<ModelRegistry>,
    script: FakeScript,
    loads: Loads,
    config: Config,
    shutdown: CancellationToken,
    dir: PathBuf,
}

/// The config of models `names`, whose files exist in `dir`.
fn config_of(dir: &Path, names: &[&str]) -> Config {
    let models = names
        .iter()
        .map(|n| {
            let path = dir.join(format!("{n}.onnx"));
            std::fs::write(&path, b"fake").unwrap();
            ModelConfig {
                name: Some(n.to_string()),
                path,
                family: ModelFamilyKind::Yolo26,
                ..ModelConfig::default()
            }
        })
        .collect();
    Config {
        models,
        download_dir: Some(dir.to_path_buf()),
        ..Config::default()
    }
}

impl Fixture {
    /// A registry over models `names` on the candidates `gpu` then `cpu`, not started until
    /// `prepare` has scripted its loads.
    fn start(names: &[&str], prepare: impl FnOnce(&Loads)) -> Self {
        Self::start_after(names, None, prepare)
    }

    fn start_after(
        names: &[&str],
        previous: Option<Weak<ModelRegistry>>,
        prepare: impl FnOnce(&Loads),
    ) -> Self {
        let dir = std::env::temp_dir().join(format!("bop_reload_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = config_of(&dir, names);
        let loads = Loads::default();
        prepare(&loads);
        let script = FakeScript::new(1, (64, 64));
        let rt = Runtimes::fake(script.clone());
        let startup = StartupDevices::of(&rt);
        let plans = plan_models(&config, &rt, None, DeviceSpec::Device(cpu())).unwrap();
        let shutdown = CancellationToken::new();
        let probe = loads.clone();
        let faults = TestFaults {
            candidates: Some(vec![
                Candidate {
                    device: gpu(),
                    fell_back: false,
                    note: None,
                },
                Candidate {
                    device: cpu(),
                    fell_back: true,
                    note: None,
                },
            ]),
            load: Some(Arc::new(move |c: &Candidate| probe.at(c))),
            ..TestFaults::default()
        };
        let spawner = Spawner {
            runtimes: Arc::new(Mutex::new(rt)),
            metrics: Arc::new(Metrics::new("0")),
            shutdown: shutdown.clone(),
            profile: config.profile(),
            prep: crate::prep::PrepPool::new(1),
            peers: Peers::new(),
            prep_threads: 1,
            best_cpu: DeviceSpec::Device(cpu()),
            can_use_gpu: false,
            provision: RwLock::new(None),
            reload: Mutex::new(()),
            previous,
            loaded: LoadedRuntimes::default(),
            startup,
            faults,
        };
        let reg = Arc::new(ModelRegistry::with_spawner(
            &config,
            spawner,
            plans,
            RuntimeInfo::default(),
        ));
        Self {
            reg,
            script,
            loads,
            config,
            shutdown,
            dir,
        }
    }

    fn handle(&self, name: &str) -> Arc<WorkerHandle> {
        self.reg.by_name(name).unwrap()
    }

    fn ready(&self, name: &str) -> Arc<WorkerHandle> {
        wait_until("the model to be ready", || {
            self.reg.by_name(name).is_some_and(|w| w.state.is_ready())
        });
        self.handle(name)
    }

    /// `name` with its inference broken for good: `Failed`, marked stuck, engine gone.
    fn break_model(&self, name: &str) -> Arc<WorkerHandle> {
        let w = self.ready(name);
        self.script.push([Step::Panic]);
        assert!(!detect(&w).success);
        wait_until("the model to be marked stuck", || w.is_stuck());
        w
    }

    fn reload(&self, name: &str) -> Reload {
        self.reg.reload_model(name, &self.config, None).unwrap()
    }

    fn stop(self) {
        self.shutdown.cancel();
        self.reg.shutdown();
        std::fs::remove_dir_all(&self.dir).ok();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // A failing test must not leave a load blocked.
        self.loads.release();
    }
}

#[test]
fn a_config_reload_serves_on_the_old_worker_until_the_new_one_is_ready() {
    let f = Fixture::start(&["a", "b"], |_| {});
    let old = f.ready("a");
    f.ready("b");
    assert_eq!(old.loaded_device().as_deref(), Some("openvino:gpu"));
    let peers = f.reg.peers();

    f.loads.hold_after(f.loads.tried().len());
    let r = f.reload("a");
    assert!(matches!(r, Reload::Spawned { replaced: true, .. }), "{r:?}");
    wait_until("the replacement to reach its compile", || {
        f.loads.tried().len() == 3
    });

    // The whole wait is served, by the old worker.
    assert!(f.reg.is_replacing("a"));
    for _ in 0..5 {
        let w = f.handle("a");
        assert!(Arc::ptr_eq(&w, &old));
        assert_eq!(w.state.get(), ModelState::Ready);
        assert!(detect(&w).success);
    }
    assert_eq!(f.reg.reloads("a"), 0);
    // The stats stay the old worker's until the swap, then name what was asked for: as a device
    // change would show them.
    old.metrics.begin_load("the device before");

    f.loads.release();
    wait_until("the swap", || !Arc::ptr_eq(&f.handle("a"), &old));
    let new = f.handle("a");
    assert!(new.state.is_ready());
    assert!(detect(&new).success);
    assert_ne!(new.metrics.requested_device(), "the device before");
    assert_eq!(f.reg.reloads("a"), 1);
    assert_eq!(f.reg.reloads("b"), 0);
    assert!(!f.reg.is_replacing("a"));
    // The directory of workers was republished with the swap.
    assert!(Arc::ptr_eq(&peers.get("a").unwrap(), &new));
    // The old worker finishes once its handles are gone.
    let weak = Arc::downgrade(&old);
    drop(old);
    wait_until("the old handle to go", || weak.upgrade().is_none());
    assert_eq!(f.reg.names(), ["a", "b"]);
    f.stop();
}

#[test]
fn live_settings_reach_a_replacement_still_loading() {
    let mut f = Fixture::start(&["a"], |_| {});
    let old = f.ready("a");
    f.loads.hold_after(f.loads.tried().len());
    f.reload("a");
    f.config.confidence_threshold = 0.77;
    let applied = f.reg.apply_live(&f.config);
    assert_eq!(applied.models, ["a"]);
    assert_eq!(old.live().confidence_threshold, 0.77);
    f.loads.release();
    wait_until("the swap", || !Arc::ptr_eq(&f.handle("a"), &old));
    assert_eq!(f.handle("a").live().confidence_threshold, 0.77);
    f.stop();
}

#[test]
fn no_memory_beside_the_old_worker_retires_it_and_tries_again() {
    let f = Fixture::start(&["a"], |_| {});
    let old = f.ready("a");
    let weak = Arc::downgrade(&old);
    drop(old);
    let before = f.loads.tried().len();
    f.loads.fail(gpu(), &[OOM]);
    f.loads.hold_after(before + 1);
    f.reload("a");

    // Second try of the same candidate, not the fallback: the old worker is gone by then, and
    // the gap is an "initializing" model, never no model.
    wait_until("the second try", || f.loads.tried().len() == before + 2);
    assert_eq!(f.loads.tried()[before..], ["openvino:gpu", "openvino:gpu"]);
    assert!(weak.upgrade().is_none(), "the old worker was not retired");
    let w = f.handle("a");
    assert_eq!(w.state.get(), ModelState::Initializing);
    assert!(!f.reg.is_replacing("a"));
    assert_eq!(f.reg.reloads("a"), 1);

    f.loads.release();
    let w = f.ready("a");
    assert_eq!(w.loaded_device().as_deref(), Some("openvino:gpu"));
    assert!(w.better().is_none());
    assert!(detect(&w).success);
    f.stop();
}

#[test]
fn making_room_is_bounded_when_the_old_worker_cannot_finish() {
    let f = Fixture::start(&["a"], |_| {});
    let old = f.ready("a");
    // An inference that does not come back holds the old worker in its slot.
    f.script.push([Step::Hang]);
    let (tx, _keep) = tokio::sync::oneshot::channel();
    let req = VisionDetectionRequest {
        image_data: jpeg().into(),
        image_name: "x.jpg".into(),
        ..VisionDetectionRequest::default()
    };
    old.sender
        .try_send(WorkItem::Detect(req, tx, Instant::now()))
        .unwrap();
    wait_until("the inference to start", || f.script.submits() == 1);
    drop(old);
    let before = f.loads.tried().len();
    f.loads.fail(gpu(), &[OOM]);
    let t = Instant::now();
    f.reload("a");
    wait_until("the second try", || f.loads.tried().len() == before + 2);
    assert!(t.elapsed() < FREE_WAIT + SOON, "{:?}", t.elapsed());
    f.ready("a");
    f.script.release(Ok(Vec::new()));
    f.stop();
}

#[test]
fn a_replacement_that_loads_nowhere_takes_over_so_its_error_shows() {
    let f = Fixture::start(&["a"], |_| {});
    let old = f.ready("a");
    f.loads.fail(gpu(), &[NO_OP]);
    f.loads.fail(cpu(), &[NO_OP]);
    f.reload("a");
    wait_until("the swap", || !Arc::ptr_eq(&f.handle("a"), &old));
    let w = f.handle("a");
    assert!(matches!(w.state.get(), ModelState::Failed(m) if m.contains("Unsupported")));
    // A config's failure: not retried.
    assert_eq!(Status::of(&w), Status::Other);
    f.stop();
}

#[test]
fn a_failed_model_is_gone_before_its_replacement_loads() {
    let f = Fixture::start(&["a"], |_| {});
    let old = f.break_model("a");
    let before = f.loads.tried().len();
    // What the probe saw when the new worker's load began, and how long after the reload: None
    // until it ran. The loader records the try before it runs the probe, so waiting on the try
    // alone could read the flag too soon (how main's CI failed twice before 242bdfc).
    let seen: Arc<Mutex<Option<(bool, Duration)>>> = Arc::new(Mutex::new(None));
    let start: Arc<std::sync::OnceLock<Instant>> = Arc::default();
    let (probe_old, slot, since) = (old.clone(), seen.clone(), start.clone());
    f.loads.set_probe(move |_| {
        let gone = thread_gone(&probe_old);
        let waited = since.get().map_or(Duration::ZERO, Instant::elapsed);
        *slot.lock_recover() = Some((gone, waited));
    });
    f.loads.hold_after(before + 1);
    start.get_or_init(Instant::now);
    f.reg.recover_model("a", &f.config).unwrap();
    // Swapped at once: it was not serving, so nothing is kept for the load.
    let w = f.handle("a");
    assert!(!Arc::ptr_eq(&w, &old));
    wait_until("the new worker's load to look", || {
        seen.lock_recover().is_some()
    });
    let (gone, waited) = seen.lock_recover().unwrap_or_default();
    // The contract (`replace_at_once`): the load waits until the old thread ended, at most
    // `FREE_WAIT`. An old thread still running is a failure only before that bound ran out, so
    // an old worker starved past it on a loaded runner is not one.
    assert!(
        gone || waited >= FREE_WAIT,
        "old thread still running {waited:?} after the reload (the load must wait up to \
         {FREE_WAIT:?})"
    );
    f.loads.release();
    f.ready("a");
    f.stop();
}

#[test]
fn a_reload_that_loads_nowhere_keeps_the_model_on_the_supervisors_list() {
    let f = Fixture::start(&["a"], |_| {});
    f.break_model("a");
    f.loads.fail(gpu(), &[LOST]);
    f.loads.fail(cpu(), &[LOST]);
    f.reg.recover_model("a", &f.config).unwrap();
    wait_until("the reload to fail", || {
        matches!(f.handle("a").state.get(), ModelState::Failed(_))
    });
    let w = f.handle("a");
    assert!(!w.is_stuck());
    assert_eq!(Status::of(&w), Status::ReloadFailed);
    assert_eq!(w.device_before_fault().as_deref(), Some("openvino:gpu"));

    // Nothing kept the failed one: the next automatic reload works and serves.
    f.reg.recover_model("a", &f.config).unwrap();
    let w = f.ready("a");
    assert_eq!(Status::of(&w), Status::Ready);
    assert!(detect(&w).success);

    // A saved config that fails is not retried: the operator has it.
    f.break_model("a");
    f.loads.fail(gpu(), &[NO_OP]);
    f.loads.fail(cpu(), &[NO_OP]);
    f.reload("a");
    wait_until("the reload to fail", || {
        matches!(f.handle("a").state.get(), ModelState::Failed(_))
    });
    assert_eq!(Status::of(&f.handle("a")), Status::Other);
    f.stop();
}

#[test]
fn a_disable_while_a_reload_loads_drops_the_reload() {
    let mut f = Fixture::start(&["a", "b"], |_| {});
    let old = f.ready("a");
    f.ready("b");
    f.loads.hold_after(f.loads.tried().len());
    f.reload("a");
    wait_until("the reload to start", || f.reg.is_replacing("a"));
    f.config.models[0].enabled = false;
    let r = f.reload("a");
    assert_eq!(r, Reload::Stopped { name: "a".into() });
    assert!(!f.reg.is_replacing("a"));
    assert!(f.reg.by_name("a").is_none());
    f.loads.release();
    drop(old);
    // Nothing brings it back when the load ends.
    f.ready("b");
    assert_eq!(f.reg.names(), ["b"]);
    assert_eq!(f.reg.reloads("a"), 0);
    f.stop();
}

#[test]
fn a_second_reload_replaces_the_one_loading() {
    let f = Fixture::start(&["a"], |_| {});
    let old = f.ready("a");
    let n = f.loads.tried().len();
    f.loads.hold_after(n);
    f.reload("a");
    wait_until("the first reload to compile", || {
        f.loads.tried().len() == n + 1
    });
    f.reload("a");
    wait_until("the second reload to compile", || {
        f.loads.tried().len() == n + 2
    });
    assert!(Arc::ptr_eq(&f.handle("a"), &old));
    f.loads.release();
    wait_until("the swap", || !Arc::ptr_eq(&f.handle("a"), &old));
    wait_until("the reload to end", || !f.reg.is_replacing("a"));
    assert_eq!(f.reg.reloads("a"), 1);
    f.stop();
}

#[test]
fn a_reload_of_a_lazy_model_swaps_at_once() {
    let mut f = Fixture::start(&["a"], |_| {});
    let old = f.ready("a");
    f.config.models[0].lazy = true;
    f.reload("a");
    let w = f.handle("a");
    assert!(!Arc::ptr_eq(&w, &old));
    assert!(w.accepts_while_initializing());
    assert!(!f.reg.is_replacing("a"));
    f.stop();
}

#[test]
fn shutdown_with_a_replacement_loading_returns() {
    let f = Fixture::start(&["a"], |_| {});
    f.ready("a");
    f.loads.hold_after(f.loads.tried().len());
    f.reload("a");
    wait_until("the reload to start", || f.reg.is_replacing("a"));
    f.loads.release();
    f.stop();
}

#[test]
fn a_fallback_for_lack_of_memory_records_the_better_device() {
    let f = Fixture::start(&["a"], |l| l.fail(gpu(), &[OOM]));
    let w = f.ready("a");
    assert_eq!(w.loaded_device().as_deref(), Some("openvino:cpu"));
    let b = w.better().expect("a better device");
    assert_eq!(b.candidate.device, gpu());
    assert_eq!(b.failure, LoadFailure::Memory);
    assert!(b.reason.contains("CL_OUT_OF_RESOURCES"), "{}", b.reason);
    f.stop();

    let f = Fixture::start(&["a"], |l| l.fail(gpu(), &[LOST]));
    let b = f.ready("a").better().expect("a better device");
    assert_eq!(b.failure, LoadFailure::DeviceLost);
    f.stop();
}

#[test]
fn a_fallback_for_any_other_reason_is_final() {
    let f = Fixture::start(&["a"], |l| l.fail(gpu(), &[NO_OP]));
    let w = f.ready("a");
    assert_eq!(w.loaded_device().as_deref(), Some("openvino:cpu"));
    assert!(w.better().is_none());
    f.stop();
}

#[test]
fn a_failed_try_of_the_better_device_leaves_the_worker_alone() {
    let f = Fixture::start(&["a"], |l| l.fail(gpu(), &[OOM]));
    let w = f.ready("a");
    let b = w.better().unwrap();
    let n = f.loads.tried().len();

    // Still no memory: the worker stays, tries go on.
    f.loads.fail(gpu(), &[OOM]);
    assert!(f.reg.retry_better("a", &f.config, &b).unwrap());
    wait_until("the try to end", || !f.reg.is_replacing("a"));
    assert_eq!(f.loads.tried()[n..], ["openvino:gpu"], "no fallback");
    assert!(Arc::ptr_eq(&f.handle("a"), &w));
    assert!(w.state.is_ready());
    assert!(w.better().is_some());
    assert!(detect(&w).success);
    assert_eq!(f.reg.reloads("a"), 0);

    // It fails for a reason that will not pass: no more tries.
    f.loads.fail(gpu(), &[NO_OP]);
    assert!(f.reg.retry_better("a", &f.config, &b).unwrap());
    wait_until("the try to end", || !f.reg.is_replacing("a"));
    assert!(Arc::ptr_eq(&f.handle("a"), &w));
    assert!(w.better().is_none());
    f.stop();
}

#[test]
fn a_try_of_the_better_device_that_loads_takes_over() {
    let f = Fixture::start(&["a"], |l| l.fail(gpu(), &[OOM]));
    let w = f.ready("a");
    let b = w.better().unwrap();
    assert!(f.reg.retry_better("a", &f.config, &b).unwrap());
    wait_until("the swap", || !Arc::ptr_eq(&f.handle("a"), &w));
    let new = f.handle("a");
    assert_eq!(new.loaded_device().as_deref(), Some("openvino:gpu"));
    assert!(new.better().is_none());
    assert!(detect(&new).success);
    assert_eq!(f.reg.reloads("a"), 1);
    f.stop();
}

#[test]
fn a_config_change_wins_over_a_try_of_the_better_device() {
    let f = Fixture::start(&["a"], |l| l.fail(gpu(), &[OOM]));
    let w = f.ready("a");
    let b = w.better().unwrap();
    let n = f.loads.tried().len();
    f.loads.hold_after(n);
    assert!(f.reg.retry_better("a", &f.config, &b).unwrap());
    // One try at a time.
    assert!(!f.reg.retry_better("a", &f.config, &b).unwrap());
    wait_until("the try to compile", || f.loads.tried().len() == n + 1);
    f.reload("a");
    f.loads.release();
    wait_until("the config reload to take over", || {
        !Arc::ptr_eq(&f.handle("a"), &w)
    });
    wait_until("the reload to end", || !f.reg.is_replacing("a"));
    assert_eq!(f.reg.reloads("a"), 1);
    f.stop();
}

#[test]
fn a_new_generation_with_no_memory_asks_the_old_one_to_give_way() {
    let old = Fixture::start(&["a", "b"], |_| {});
    old.ready("a");
    old.ready("b");
    let old_a = old.handle("a");
    let weak = Arc::downgrade(&old_a);
    drop(old_a);
    let new = Fixture::start_after(&["a", "b"], Some(Arc::downgrade(&old.reg)), |l| {
        l.fail(gpu(), &[OOM]);
    });
    let w = new.ready("a");
    new.ready("b");
    // Second try of the first candidate, no fallback; only that model gave way.
    assert_eq!(w.loaded_device().as_deref(), Some("openvino:gpu"));
    assert!(w.better().is_none());
    assert_eq!(new.loads.tried()[..2], ["openvino:gpu", "openvino:gpu"]);
    assert!(weak.upgrade().is_none());
    let stand_in = old.handle("a");
    assert_eq!(stand_in.state.get(), ModelState::Initializing);
    assert!(
        stand_in.state.detail().unwrap().contains("taking over"),
        "{:?}",
        stand_in.state.detail()
    );
    assert!(detect(&old.handle("b")).success);
    new.stop();
    old.stop();
}

#[test]
fn nothing_to_give_way_means_no_second_try() {
    let old = Fixture::start(&["b"], |_| {});
    old.ready("b");
    let new = Fixture::start_after(&["a"], Some(Arc::downgrade(&old.reg)), |l| {
        l.fail(gpu(), &[OOM]);
    });
    let w = new.ready("a");
    assert_eq!(w.loaded_device().as_deref(), Some("openvino:cpu"));
    assert!(w.better().is_some());
    assert_eq!(new.loads.tried(), ["openvino:gpu", "openvino:cpu"]);
    new.stop();
    old.stop();
}

/// The runner's order after the takeover wait: a save that landed on the old generation
/// reaches the new one before it serves, and a model disabled meanwhile never answers there.
#[test]
fn a_save_during_the_takeover_wait_reaches_the_new_generation() {
    let f = Fixture::start(&["a", "b"], |_| {});
    f.ready("a");
    f.ready("b");
    let state = Arc::new(crate::server::AppState::new(
        f.reg.clone(),
        Arc::new(Metrics::new("0")),
        f.config.clone(),
        f.dir.join("cfg.json"),
    ));
    let based_on = state.config.revision();
    state
        .config
        .update("test", |c| {
            c.models[1].enabled = false;
            c.confidence_threshold = 0.66;
            Ok(())
        })
        .unwrap();
    let stale = f.config.clone();
    assert!(
        state
            .config
            .adopt(Some(stale), "restart (config file reloaded)", based_on)
            .is_none(),
        "the save must win"
    );
    crate::server::apply_config(&state);
    assert!(f.reg.by_name("b").is_none());
    assert_eq!(f.handle("a").live().confidence_threshold, 0.66);
    drop(state);
    f.stop();
}
