//! The stop timeline with fake targets (call order and times), the forced-exit child filter, the
//! orphan sweep, and the service-manager stop budgets read from the repository's deploy files.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A fake target: every call is recorded with its time; the step-1 hooks sleep.
#[derive(Default)]
struct Fake {
    calls: Mutex<Vec<(&'static str, Instant)>>,
    /// How long each step-1 hook takes; the collector's is `collector_takes`.
    hook_takes: Duration,
    collector_takes: Duration,
    benchmark_deadline: Mutex<Option<Instant>>,
    hooks_done: AtomicUsize,
}

impl Fake {
    fn note(&self, what: &'static str) {
        self.calls.lock().unwrap().push((what, Instant::now()));
    }
    fn hook(&self, start: &'static str, end: &'static str, takes: Duration) {
        self.note(start);
        std::thread::sleep(takes);
        self.note(end);
        self.hooks_done.fetch_add(1, Ordering::SeqCst);
    }
    fn at(&self, what: &str) -> Instant {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(w, _)| *w == what)
            .map(|(_, t)| *t)
            .unwrap_or_else(|| panic!("{what} was not called"))
    }
    fn count(&self, what: &str) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|(w, _)| *w == what)
            .count()
    }
}

impl StopTargets for Fake {
    fn stop_training(&self, _: Instant) {
        self.hook("training", "training done", self.hook_takes);
    }
    fn stop_provisioner(&self, _: Instant) {
        self.hook("provisioner", "provisioner done", self.hook_takes);
    }
    fn stop_collector(&self, _: Instant) {
        self.hook("collector", "collector done", self.collector_takes);
    }
    fn cancel_benchmark(&self) {
        self.note("cancel benchmark");
    }
    fn wait_benchmark(&self, deadline: Instant) -> bool {
        self.note("wait benchmark");
        *self.benchmark_deadline.lock().unwrap() = Some(deadline);
        true
    }
    fn flush_auth(&self) {
        self.note("flush auth");
    }
}

fn fake(hook_takes: Duration, collector_takes: Duration) -> Arc<Fake> {
    Arc::new(Fake {
        hook_takes,
        collector_takes,
        ..Fake::default()
    })
}

/// Every step-1 hook is signalled at t0 and they run side by side, before any wait; the waits
/// come in order (HTTP, benchmark, the joins, the auth flush, the retire), and the whole stop
/// takes about the longest hook, not the sum.
#[test]
fn step_one_is_signalled_together_and_the_waits_follow_in_order() {
    let hook = Duration::from_millis(300);
    let f = fake(hook, hook);
    let tl = Timeline::with_budget(
        f.clone(),
        Duration::from_secs(5),
        Duration::from_secs(5),
        Duration::from_secs(1),
    );
    let t0 = tl.begin();
    // Idempotent: the same t0, nothing signalled twice.
    assert_eq!(tl.begin(), t0);
    let http_deadline = Mutex::new(None);
    let report = tl.finish(
        |d| {
            f.note("http");
            *http_deadline.lock().unwrap() = Some(d);
        },
        || f.note("retire"),
    );
    assert_eq!(report.abandoned, []);
    assert!(report.benchmark_idle);
    for what in ["training", "provisioner", "collector", "cancel benchmark"] {
        assert_eq!(f.count(what), 1, "{what}");
    }
    // Signalled at t0, together: every hook started before any of them ended (a serial stop
    // would end the first before the next one starts), and the benchmark cancel did not wait.
    let first_end = ["training done", "provisioner done", "collector done"]
        .iter()
        .map(|w| f.at(w))
        .min()
        .unwrap();
    for start in ["training", "provisioner", "collector"] {
        assert!(
            f.at(start) < first_end,
            "{start} started after a hook ended"
        );
        assert!(f.at(start) - t0 < Duration::from_millis(200), "{start}");
    }
    assert!(f.at("cancel benchmark") - t0 < Duration::from_millis(100));
    // The waits in order, each measured from t0.
    assert!(f.at("http") < f.at("wait benchmark"));
    assert!(f.at("wait benchmark") < f.at("flush auth"));
    assert!(f.at("collector done") <= f.at("flush auth"), "joined first");
    assert!(f.at("training done") <= f.at("flush auth"), "joined first");
    assert!(f.at("flush auth") < f.at("retire"));
    assert_eq!(
        *http_deadline.lock().unwrap(),
        Some(t0 + crate::runner::SERVER_STOP_TIMEOUT)
    );
    assert_eq!(
        *f.benchmark_deadline.lock().unwrap(),
        Some(t0 + Duration::from_secs(5))
    );
    assert!(
        report.took < hook * 2,
        "hooks ran one after another: {:?}",
        report.took
    );
}

/// A hook that overruns its deadline does not hold the stop: it is abandoned at the join
/// deadline (named in the report) and the rest still runs.
#[test]
fn an_overrunning_hook_is_abandoned_at_its_deadline() {
    let f = fake(Duration::from_millis(10), Duration::from_secs(3));
    let tl = Timeline::with_budget(
        f.clone(),
        Duration::from_millis(200),
        Duration::from_millis(200),
        Duration::from_millis(100),
    );
    let started = Instant::now();
    let report = tl.finish(|_| {}, || f.note("retire"));
    assert_eq!(report.abandoned, [Hook::Collector]);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(f.count("retire"), 1);
    assert_eq!(f.count("flush auth"), 1);
}

/// `finish` without a stop request (the run ended on an error) still stops everything.
#[test]
fn finish_alone_begins_the_stop() {
    let f = fake(Duration::ZERO, Duration::ZERO);
    let tl = Timeline::new(f.clone());
    tl.finish(|_| {}, || {});
    assert_eq!(f.hooks_done.load(Ordering::SeqCst), 3);
    assert_eq!(f.count("cancel benchmark"), 1);
}

/// `arm` starts the stop the moment the token is cancelled (nobody calls `begin`), and its task
/// lets go of the timeline when the run ends without a stop request.
#[test]
fn arm_begins_on_the_stop_request_and_ends_with_the_run() {
    let rt = crate::runner::build_runtime().unwrap();
    let f = fake(Duration::ZERO, Duration::ZERO);
    let tl = Timeline::new(f.clone());
    let token = CancellationToken::new();
    arm(&rt, &tl, &token);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(f.count("cancel benchmark"), 0, "nothing before the request");
    token.cancel();
    let deadline = Instant::now() + Duration::from_secs(5);
    while f.hooks_done.load(Ordering::SeqCst) < 3 {
        assert!(Instant::now() < deadline, "the stop did not begin");
        std::thread::sleep(Duration::from_millis(10));
    }
    tl.finish(|_| {}, || {});

    // A run that ends with no stop request: the task ends with it.
    let tl = Timeline::new(fake(Duration::ZERO, Duration::ZERO));
    arm(&rt, &tl, &CancellationToken::new());
    assert_eq!(Arc::strong_count(&tl), 2, "the task holds it");
    tl.finish(|_| {}, || {});
    let deadline = Instant::now() + Duration::from_secs(5);
    while Arc::strong_count(&tl) > 1 {
        assert!(Instant::now() < deadline, "the armed task outlived the run");
        std::thread::sleep(Duration::from_millis(10));
    }
    rt.shutdown_timeout(Duration::from_secs(1));
}

/// A fake OS view: (pid -> (start time, exe)).
struct OsView(Vec<(u32, u64, &'static str)>);

impl Probe for OsView {
    fn start_time(&self, pid: u32) -> Option<u64> {
        self.0.iter().find(|p| p.0 == pid).map(|p| p.1)
    }
    fn exe_path(&self, pid: u32) -> Option<PathBuf> {
        self.0
            .iter()
            .find(|p| p.0 == pid)
            .map(|p| PathBuf::from(p.2))
    }
}

fn entry(pid: u32, start: u64, exe: &str, owner: (u32, u64)) -> Entry {
    Entry {
        pid,
        start_time: start,
        exe: exe.into(),
        kind: "export".into(),
        started_at: "2026-11-22T00:00:00Z".into(),
        owner_pid: owner.0,
        owner_start_time: owner.1,
    }
}

/// The forced exit kills only this process's own children that are provably still the same
/// processes.
#[test]
fn a_forced_exit_kills_only_own_children_that_still_match() {
    let me = (100, 7);
    let os = OsView(vec![
        (10, 1, "/py"),
        (11, 2, "/py"),
        (12, 99, "/py"),
        (13, 4, "/other"),
        (14, 5, "/py"),
    ]);
    let entries = vec![
        entry(10, 1, "/py", me),       // ours, alive, matches: killed
        entry(11, 2, "/py", (200, 7)), // another process's
        entry(12, 3, "/py", me),       // PID reused (start time differs)
        entry(13, 4, "/py", me),       // exe differs
        entry(14, 5, "/py", (100, 8)), // an earlier process with our PID
        entry(15, 6, "/py", me),       // gone
    ];
    let kept: Vec<u32> = own_live(entries.clone(), me.0, Some(me.1), &os)
        .iter()
        .map(|e| e.pid)
        .collect();
    assert_eq!(kept, [10]);
    // Our own start time unreadable: nothing is provably ours.
    assert!(own_live(entries, me.0, None, &os).is_empty());
}

/// A real child recorded by this process is killed by the forced-exit path, and nothing else
/// in the registry is touched.
#[cfg(unix)]
#[test]
fn kill_own_in_ends_a_recorded_child() {
    let root = std::env::temp_dir().join(format!("bop_stop_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let mut cmd = std::process::Command::new("sleep");
    cmd.arg("30");
    crate::backend::process::prepare(&mut cmd);
    let mut child = cmd.spawn().unwrap();
    let recorded = Registry::new(&root).record("test", child.id()).unwrap();
    assert_eq!(kill_own_in(std::slice::from_ref(&root)), 1);
    let status = child.wait().unwrap();
    assert!(!status.success(), "{status:?}");
    drop(recorded);
    std::fs::remove_dir_all(&root).ok();
}

/// The orphan sweep runs once per data root and process: a second call does not sweep again.
#[test]
fn the_orphan_sweep_runs_once_per_data_root() {
    let root = std::env::temp_dir().join(format!("bop_sweep_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    // A stale entry: its owner and the child are gone (start times nobody has).
    let stale = || {
        let file = serde_json::json!({
            "format": crate::children::FORMAT,
            "children": [entry(u32::MAX - 1, 1, "/nope", (u32::MAX - 2, 1))],
        });
        std::fs::write(root.join(crate::children::FILE), file.to_string()).unwrap();
    };
    stale();
    sweep_orphans_once(&root);
    assert!(Registry::new(&root).entries().unwrap().is_empty());
    stale();
    sweep_orphans_once(&root);
    assert_eq!(
        Registry::new(&root).entries().unwrap().len(),
        1,
        "swept once"
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn an_abandoned_benchmark_names_its_device() {
    let running = crate::benchmark::service::Running {
        models: vec![],
        progress: Some(crate::benchmark::service::ProgressView {
            device: "openvino:gpu.0".into(),
            ..Default::default()
        }),
        started: Some(Instant::now()),
    };
    let msg = abandoned_benchmark(Some(&running), Duration::from_secs(15));
    assert!(
        msg.contains("abandoned on openvino:gpu.0 after 15 s"),
        "{msg}"
    );
    assert!(msg.contains("breadcrumb stays"), "{msg}");
    let msg = abandoned_benchmark(None, Duration::from_secs(30));
    assert!(msg.contains("abandoned on its device after 30 s"), "{msg}");
}

/// The repository file `rel` (deploy files, scripts, sources).
fn repo_file(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The integer after `key` on the first line (or the next line, for a plist) holding it.
fn number_after(text: &str, key: &str) -> u64 {
    let at = text.find(key).unwrap_or_else(|| panic!("{key} not found")) + key.len();
    let digits: String = text
        .get(at..)
        .unwrap()
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("no number after {key}"))
}

/// RA-026 / QA-075: every service manager waits longer than the app's own hard bound, so the
/// app's bounded end always runs first, and the scripts that wait for launchd wait as long as
/// launchd may take.
#[test]
fn service_manager_stop_budgets_cover_the_app_s_hard_bound() {
    use crate::runner::{SERVICE_CTL_STOP_WAIT, SERVICE_MANAGER_STOP, SHUTDOWN_DEADLINE};
    let want = SERVICE_MANAGER_STOP.as_secs();
    assert!(want >= SHUTDOWN_DEADLINE.as_secs() + 15);

    let unit = repo_file("deploy/blue-onyx-prism.service");
    assert_eq!(number_after(&unit, "TimeoutStopSec="), want);
    assert!(
        unit.lines().any(|l| l.trim() == "KillMode=control-group"),
        "the unit keeps KillMode=control-group: the cgroup kill is the backstop"
    );
    let plist = repo_file("deploy/com.blueonyx.prism.plist");
    assert_eq!(number_after(&plist, "<key>ExitTimeOut</key>"), want);
    let compose = repo_file("deploy/docker-compose.yml");
    assert_eq!(number_after(&compose, "stop_grace_period:"), want);
    // Orphans of killed job trees are reaped in the container (PID 1 is an init).
    let dockerfile = repo_file("Dockerfile");
    assert!(
        dockerfile.contains(
            r#"ENTRYPOINT ["/usr/bin/tini", "-s", "--", "/bin/sh", "/app/docker-entrypoint.sh"]"#
        ),
        "the image's entrypoint is tini"
    );
    // The launchd scripts wait for bootout at least as long as launchd may take.
    for script in ["scripts/launchd_common.sh", "scripts/uninstall_launchd.sh"] {
        let text = repo_file(script);
        assert!(
            number_after(&text, "$(seq 1 ") >= want + 5,
            "{script} waits less than ExitTimeOut"
        );
    }
    // `service stop|restart` waits longer than the service manager (macOS and Windows use these
    // constants; systemctl blocks on the job itself).
    assert!(SERVICE_CTL_STOP_WAIT > SERVICE_MANAGER_STOP);
    let mac = repo_file("src/service_ctl/macos.rs");
    assert!(mac.contains("const STOP_TIMEOUT: Duration = crate::runner::SERVICE_CTL_STOP_WAIT;"));
    let win = repo_file("src/service_ctl/windows.rs");
    assert!(win.contains("const STOP_TIMEOUT: Duration = crate::runner::SERVICE_CTL_STOP_WAIT;"));
    // The Windows installers set the preshutdown timeout through the service binary.
    let ps1 = repo_file("scripts/install_service.ps1");
    assert!(ps1.contains("--set-preshutdown-timeout"));
}
