//! Tests of the training queue's pause / resume and of a cancelled task that kept partial work,
//! over a fake task whose one "helper process" blocks until cancelled.

use super::*;
use crate::resources::catalog::Resource;
use crate::resources::export::{Cancelled, Cmd, Stage};
use std::sync::atomic::AtomicUsize;

/// One helper that runs until cancelled; records the pause flag it is handed.
#[derive(Default)]
struct HoldRunner {
    /// Helpers started.
    started: AtomicUsize,
    /// The last pause flag value seen by a running helper.
    paused: AtomicBool,
}

impl CommandRunner for HoldRunner {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        self.run_pausable(cmd, on_line, cancel, &AtomicBool::new(false))
    }

    fn run_pausable(
        &self,
        _cmd: &Cmd,
        on_line: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
        pause: &AtomicBool,
    ) -> Result<()> {
        self.started.fetch_add(1, Ordering::Relaxed);
        on_line("helper started");
        while !cancel.load(Ordering::Relaxed) {
            self.paused
                .store(pause.load(Ordering::Relaxed), Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(10));
        }
        Err(anyhow::anyhow!(Cancelled))
    }
}

struct NoFetch;

impl Fetcher for NoFetch {
    fn fetch(
        &self,
        _res: &'static Resource,
        _dir: &Path,
        _progress: &mut dyn FnMut(u64, u64),
        _cancel: &AtomicBool,
    ) -> Result<()> {
        anyhow::bail!("no downloads in this test")
    }
}

type Result<T> = anyhow::Result<T>;

/// A task with one helper; `keep` = on cancel it returns what it kept (`Ok`) instead of the
/// cancellation error.
struct HoldTask {
    id: String,
    keep: bool,
}

impl Task for HoldTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> task::TaskKind {
        task::TaskKind::Autolabel
    }
    fn title(&self) -> String {
        format!("hold {}", self.id)
    }
    fn steps(&self) -> usize {
        2
    }
    fn run(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<task::TaskOutcome> {
        report(TaskReport::Stage {
            stage: "labelling",
            step: 2,
            steps: 2,
            percent: 50,
            step_percent: 50,
            detail: "3 of 6 frames",
        });
        let cmd = Cmd {
            stage: Stage::Export,
            program: "helper".into(),
            args: vec![],
            env: vec![],
            cwd: PathBuf::from("."),
            timeout: Duration::from_secs(60),
            low_priority: false,
        };
        let r = ctx
            .runner
            .run(&cmd, &mut |l| report(TaskReport::Line(l)), cancel);
        match r {
            Err(_) if self.keep => Ok(task::TaskOutcome {
                summary: "kept 3 labelled frames".into(),
                result: serde_json::json!({ "frames": 3 }),
            }),
            Err(e) => Err(e),
            Ok(()) => Ok(task::TaskOutcome {
                summary: "done".into(),
                result: serde_json::Value::Null,
            }),
        }
    }
}

fn until(what: &str, mut f: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !f() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn state(svc: &TrainingService, id: &str) -> RunState {
    svc.get_task(id).map(|r| r.state).unwrap()
}

/// `cancel` reads the state, then acts on it: a job that started in between must still stop, not
/// keep running while the page says the cancel went through. Called directly on a running task,
/// which is what `cancel` meets when it loses that race.
#[test]
fn a_cancel_that_read_queued_just_before_the_start_is_not_lost() {
    let dir = std::env::temp_dir().join(format!("bop-cancel-race-{}", uuid::Uuid::new_v4()));
    let svc = TrainingService::start(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        Some(dir.join("runs.json")),
    );
    let id = svc
        .enqueue_task(Box::new(HoldTask {
            id: "t".into(),
            keep: false,
        }))
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while state(&svc, &id) != RunState::Running {
        assert!(
            std::time::Instant::now() < deadline,
            "the task never started"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(svc.cancel_queued_task(&id, None));
    let rec = svc.wait_task(&id, Duration::from_secs(10)).unwrap();
    assert_eq!(rec.state, RunState::Cancelled);
    assert!(
        svc.inner.cancelled.lock_recover().is_empty(),
        "a stale entry would cancel a later job of that id"
    );
    svc.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_running_task_pauses_resumes_keeps_its_place_and_cancels_while_paused() {
    let dir = std::env::temp_dir().join(format!("bop-train-pause-{}", uuid::Uuid::new_v4()));
    let runner = Arc::new(HoldRunner::default());
    let svc = TrainingService::start(
        runner.clone(),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        Some(dir.join("runs.json")),
    );
    let first = svc
        .enqueue_task(Box::new(HoldTask {
            id: "first".into(),
            keep: true,
        }))
        .unwrap();
    let second = svc
        .enqueue_task(Box::new(HoldTask {
            id: "second".into(),
            keep: false,
        }))
        .unwrap();
    until("the first helper", || {
        runner.started.load(Ordering::Relaxed) == 1
    });
    assert_eq!(state(&svc, &first), RunState::Running);
    assert!(!svc.pause(&second), "a queued job is not paused");
    assert!(!svc.resume(&first), "a running job is not resumed");
    assert!(!svc.pause("nope"));

    assert!(svc.pause(&first));
    assert!(!svc.pause(&first), "already paused");
    let r = svc.get_task(&first).unwrap();
    assert_eq!(r.state, RunState::Paused);
    assert!(r.state.is_busy() && r.state.is_active());
    assert_eq!(r.describe(), "paused (labelling 2/2): 3 of 6 frames");
    // The Jobs tab's clock stands still while paused, and is only the running job's.
    let t = svc.timing(&first).expect("the running job has a clock");
    assert!(svc.timing(&second).is_none());
    std::thread::sleep(Duration::from_millis(30));
    let later = svc.timing(&first).unwrap();
    assert_eq!(later.elapsed_s, t.elapsed_s);
    assert!(later.paused_s > t.paused_s);
    until("the helper to see the pause", || {
        runner.paused.load(Ordering::Relaxed)
    });
    // Persisted as paused (a restart turns it into the interrupted failure); a reader sees it
    // paused while this queue owns the folder, never interrupted.
    let raw = std::fs::read_to_string(dir.join(task::TASKS_FILE)).unwrap();
    assert!(raw.contains(r#""state": "paused""#), "{raw}");
    let back = task::load_records(&dir.join(task::TASKS_FILE));
    assert_eq!(back[0].state, RunState::Paused);
    assert_eq!(
        std::fs::read_to_string(dir.join(task::TASKS_FILE)).unwrap(),
        raw
    );
    // The paused job keeps its place.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(state(&svc, &second), RunState::Queued);
    assert_eq!(runner.started.load(Ordering::Relaxed), 1);
    assert!(svc.busy());
    assert!(svc.remove(&first).is_err(), "a paused job is not removed");

    assert!(svc.resume(&first));
    assert_eq!(state(&svc, &first), RunState::Running);
    until("the helper to see the resume", || {
        !runner.paused.load(Ordering::Relaxed)
    });

    // Cancel while paused: the task kept partial work, so it is Cancelled with its summary.
    assert!(svc.pause(&first));
    assert!(svc.cancel(&first));
    assert!(!svc.resume(&first), "a cancelling job is not resumed");
    let r = svc.wait_task(&first, Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Cancelled);
    assert_eq!(r.summary.as_deref(), Some("kept 3 labelled frames"));
    assert_eq!(r.result, Some(serde_json::json!({ "frames": 3 })));
    assert_eq!(r.describe(), "cancelled: kept 3 labelled frames");

    // The next job starts; one that returns the cancellation error is plainly cancelled.
    until("the second helper", || {
        runner.started.load(Ordering::Relaxed) == 2
    });
    assert!(!runner.paused.load(Ordering::Relaxed), "a fresh pause flag");
    assert!(svc.cancel(&second));
    let r = svc.wait_task(&second, Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Cancelled);
    assert_eq!(r.summary, None);
    assert_eq!(r.describe(), "cancelled");
    assert!(!svc.pause(&second), "a finished job is not paused");
    svc.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn paused_serializes_as_paused() {
    assert_eq!(
        serde_json::to_string(&RunState::Paused).unwrap(),
        r#""paused""#
    );
    assert_eq!(
        serde_json::from_str::<RunState>(r#""paused""#).unwrap(),
        RunState::Paused
    );
    assert!(RunState::Paused.is_busy());
    assert!(!RunState::Queued.is_active());
}

/// A task that ends at once: Done, Failed, or (with `cancelled`) as a cancelled run; `after` =
/// the task it follows. Counts its runs.
struct QuickTask {
    id: String,
    after: Option<String>,
    fail: bool,
    cancelled: bool,
    ran: Arc<AtomicUsize>,
}

impl QuickTask {
    fn new(id: &str, after: Option<&str>, ran: &Arc<AtomicUsize>) -> Self {
        Self {
            id: id.into(),
            after: after.map(str::to_string),
            fail: false,
            cancelled: false,
            ran: ran.clone(),
        }
    }
}

impl Task for QuickTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> task::TaskKind {
        if self.after.is_some() {
            task::TaskKind::Autolabel
        } else {
            task::TaskKind::Import
        }
    }
    fn title(&self) -> String {
        format!("quick {}", self.id)
    }
    fn steps(&self) -> usize {
        1
    }
    fn after(&self) -> Option<&str> {
        self.after.as_deref()
    }
    fn run(
        &self,
        _ctx: TaskContext<'_>,
        _report: &mut dyn FnMut(TaskReport<'_>),
        _cancel: &AtomicBool,
    ) -> Result<task::TaskOutcome> {
        self.ran.fetch_add(1, Ordering::Relaxed);
        if self.cancelled {
            return Err(anyhow::anyhow!(Cancelled));
        }
        if self.fail {
            anyhow::bail!("boom");
        }
        Ok(task::TaskOutcome {
            summary: format!("{} done", self.id),
            result: serde_json::json!({ "frames": 1 }),
        })
    }
}

fn quick_service() -> Arc<TrainingService> {
    TrainingService::start(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        None,
    )
}

#[test]
fn a_task_after_a_failed_or_cancelled_one_is_not_run() {
    let svc = quick_service();
    let ran = Arc::new(AtomicUsize::new(0));
    // A failed: B is Cancelled with the reason, never run.
    let mut a = QuickTask::new("a1", None, &ran);
    a.fail = true;
    svc.enqueue_task(Box::new(a)).unwrap();
    let b = QuickTask::new("b1", Some("a1"), &ran);
    let rec = TaskRecord::new(&b, String::new());
    assert_eq!(rec.after.as_deref(), Some("a1"));
    assert_eq!(rec.describe(), "queued (after a1)");
    svc.enqueue_task(Box::new(b)).unwrap();
    let r = svc.wait_task("b1", Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Cancelled);
    assert_eq!(
        r.summary.as_deref(),
        Some("not run: the import it follows failed")
    );
    assert!(r.finished.is_some());
    assert_eq!(ran.load(Ordering::Relaxed), 1, "only A ran");
    // A cancelled while running (the cancellation error): B is not run either.
    let mut a = QuickTask::new("a2", None, &ran);
    a.cancelled = true;
    svc.enqueue_task(Box::new(a)).unwrap();
    svc.enqueue_task(Box::new(QuickTask::new("b2", Some("a2"), &ran)))
        .unwrap();
    let r = svc.wait_task("b2", Duration::from_secs(10)).unwrap();
    assert_eq!(state(&svc, "a2"), RunState::Cancelled);
    assert_eq!(r.state, RunState::Cancelled);
    assert_eq!(
        r.describe(),
        "cancelled: not run: the import it follows was cancelled"
    );
    assert_eq!(ran.load(Ordering::Relaxed), 2);
    // The queue goes on.
    svc.enqueue_task(Box::new(QuickTask::new("c", None, &ran)))
        .unwrap();
    assert_eq!(
        svc.wait_task("c", Duration::from_secs(10)).unwrap().state,
        RunState::Done
    );
    svc.shutdown();
}

#[test]
fn a_task_after_a_done_or_removed_one_runs() {
    let svc = quick_service();
    let ran = Arc::new(AtomicUsize::new(0));
    svc.enqueue_task(Box::new(QuickTask::new("a", None, &ran)))
        .unwrap();
    svc.enqueue_task(Box::new(QuickTask::new("b", Some("a"), &ran)))
        .unwrap();
    let r = svc.wait_task("b", Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Done);
    assert_eq!(r.summary.as_deref(), Some("b done"));
    // The record it follows is gone (removed after it finished): it runs.
    svc.remove("a").unwrap();
    svc.enqueue_task(Box::new(QuickTask::new("c", Some("a"), &ran)))
        .unwrap();
    assert_eq!(
        svc.wait_task("c", Duration::from_secs(10)).unwrap().state,
        RunState::Done
    );
    assert_eq!(ran.load(Ordering::Relaxed), 3);
    svc.shutdown();
}

#[test]
fn cancelling_a_task_cancels_the_queued_tasks_that_follow_it() {
    let dir = std::env::temp_dir().join(format!("bop-train-after-{}", uuid::Uuid::new_v4()));
    let runner = Arc::new(HoldRunner::default());
    let svc = TrainingService::start(
        runner.clone(),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        Some(dir.join("runs.json")),
    );
    let ran = Arc::new(AtomicUsize::new(0));
    // A blocker holds the queue, so A and its followers stay queued.
    svc.enqueue_task(Box::new(HoldTask {
        id: "hold".into(),
        keep: false,
    }))
    .unwrap();
    for t in [
        QuickTask::new("a", None, &ran),
        QuickTask::new("b", Some("a"), &ran),
        QuickTask::new("c", Some("b"), &ran),
        QuickTask::new("other", None, &ran),
    ] {
        svc.enqueue_task(Box::new(t)).unwrap();
    }
    until("the blocker", || {
        runner.started.load(Ordering::Relaxed) == 1
    });
    assert!(svc.cancel("a"));
    for id in ["a", "b", "c"] {
        assert_eq!(state(&svc, id), RunState::Cancelled, "{id}");
    }
    assert_eq!(
        svc.get_task("b").unwrap().summary.as_deref(),
        Some("not run: the import it follows was cancelled")
    );
    assert_eq!(state(&svc, "other"), RunState::Queued);
    // Persisted.
    let back = task::load_records(&dir.join(task::TASKS_FILE));
    let c = back.iter().find(|r| r.id == "c").unwrap();
    assert_eq!(
        (c.state, c.after.as_deref()),
        (RunState::Cancelled, Some("b"))
    );
    assert!(svc.cancel("hold"));
    let r = svc.wait_task("other", Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Done);
    assert_eq!(
        ran.load(Ordering::Relaxed),
        1,
        "only the unrelated task ran"
    );
    svc.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn finish_is_refused_for_anything_but_a_training_run() {
    let runner = Arc::new(HoldRunner::default());
    let svc = TrainingService::start(
        runner.clone(),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        None,
    );
    assert_eq!(svc.finish("nope").unwrap_err(), "no run nope");
    let id = svc
        .enqueue_task(Box::new(HoldTask {
            id: "task".into(),
            keep: false,
        }))
        .unwrap();
    until("the helper", || runner.started.load(Ordering::Relaxed) == 1);
    let e = svc.finish(&id).unwrap_err();
    assert!(e.contains("only a training run"), "{e}");
    assert!(svc.cancel(&id));
    svc.wait_task(&id, Duration::from_secs(10)).unwrap();
    svc.shutdown();
}

#[test]
fn the_run_records_the_balance_summary_of_its_dataset() {
    let mut p = super::super::dataset::Prepared {
        dir: PathBuf::from("/runs/r1/data"),
        classes: vec!["person".into(), "raccoon".into()],
        train_images: 412,
        train_frames: 209,
        valid_images: 40,
        valid_groups: vec!["yard/2026-10-01".into()],
        heldout_dir: Some(PathBuf::from("/runs/r1/heldout")),
        heldout_images: 60,
        heldout_fingerprint: "f1".into(),
        warnings: vec!["w".into()],
        balance_summary: Some(
            "209 → 412 images/epoch; person 443→443, raccoon 32→118 boxes".into(),
        ),
    };
    let d = DatasetCounts::of(&p);
    assert_eq!(d.train_images, 412);
    assert_eq!(d.train_frames, 209);
    assert_eq!(d.valid_groups, p.valid_groups);
    assert_eq!(d.heldout_dir, p.heldout_dir);
    assert_eq!(d.balance_summary, p.balance_summary);
    let v = serde_json::to_value(&d).unwrap();
    assert_eq!(v["balance_summary"], serde_json::json!(p.balance_summary));
    // Unbalanced: no key written; older records without it read as None.
    p.balance_summary = None;
    let d = DatasetCounts::of(&p);
    let v = serde_json::to_value(&d).unwrap();
    assert!(v.get("balance_summary").is_none(), "{v}");
    let back: DatasetCounts = serde_json::from_value(v).unwrap();
    assert_eq!(back, d);
    // A record of an older version has neither field.
    let old: DatasetCounts = serde_json::from_value(serde_json::json!({
        "classes": ["person"],
        "train_images": 1019,
        "valid_images": 50,
        "heldout_images": 136,
        "heldout_dir": null,
        "warnings": [],
    }))
    .unwrap();
    assert_eq!((old.train_frames, old.valid_groups.len()), (0, 0));
}

#[test]
fn a_damaged_runs_file_is_moved_aside_before_anything_overwrites_it() {
    let dir = std::env::temp_dir().join(format!("bop-runs-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = dir.join(RUNS_FILE);
    std::fs::write(&store, b"[{\"id\":\"cut off by a cra").unwrap();
    let tasks = task::tasks_file(&store);
    std::fs::write(&tasks, b"not json").unwrap();
    let svc = TrainingService::start(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(crate::training::job::NoHooks),
        Some(store.clone()),
    );
    assert!(svc.runs().is_empty());
    let id = svc
        .enqueue_task(Box::new(HoldTask {
            id: "t1".into(),
            keep: false,
        }))
        .unwrap();
    assert!(svc.cancel(&id));
    svc.wait_task(&id, Duration::from_secs(10)).unwrap();
    svc.shutdown();
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    let aside = |n: &str| {
        names
            .iter()
            .find(|f| f.starts_with(&format!("{n}.corrupt-")))
            .unwrap_or_else(|| panic!("{n} moved aside: {names:?}"))
    };
    assert_eq!(
        std::fs::read(dir.join(aside(RUNS_FILE))).unwrap(),
        b"[{\"id\":\"cut off by a cra"
    );
    assert_eq!(
        std::fs::read(dir.join(aside(task::TASKS_FILE))).unwrap(),
        b"not json"
    );
    assert_eq!(task::load_records(&tasks).len(), 1, "the new history");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A damaged file nothing could move aside (a read-only folder that later comes back): this
/// process leaves it alone rather than save the empty history over it.
#[cfg(unix)]
#[test]
fn a_file_that_could_not_be_moved_aside_is_not_written_over() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("bop-runs-held-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = dir.join(RUNS_FILE);
    let tasks = task::tasks_file(&store);
    std::fs::write(&store, b"damaged runs").unwrap();
    std::fs::write(&tasks, b"damaged tasks").unwrap();
    let set = |mode| std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode));
    set(0o555).unwrap();
    if std::fs::File::create(dir.join("probe")).is_ok() {
        // Root ignores the mode: nothing to test here.
        let _ = set(0o755);
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let svc = TrainingService::start(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(crate::training::job::NoHooks),
        Some(store.clone()),
    );
    set(0o755).unwrap();
    let id = svc
        .enqueue_task(Box::new(HoldTask {
            id: "t1".into(),
            keep: false,
        }))
        .unwrap();
    assert!(svc.cancel(&id));
    let rec = svc.wait_task(&id, Duration::from_secs(10)).unwrap();
    assert_eq!(rec.state, RunState::Cancelled, "the queue still works");
    svc.shutdown();
    assert_eq!(std::fs::read(&store).unwrap(), b"damaged runs");
    assert_eq!(std::fs::read(&tasks).unwrap(), b"damaged tasks");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `enqueue_task_unless` decides under the task list's lock: eight starts at once of a job that
/// runs one at a time, each refused while one is busy, queue exactly one; the rest get the rule's
/// words, and the queue's own refusal (a repeated id) stays a failure.
#[test]
fn a_rule_checked_under_the_queues_lock_lets_one_of_many_concurrent_starts_through() {
    let dir = std::env::temp_dir().join(format!("bop-enqueue-unless-{}", uuid::Uuid::new_v4()));
    let svc = TrainingService::start(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        Some(dir.join("runs.json")),
    );
    let one_at_a_time = |tasks: &[TaskRecord]| {
        tasks
            .iter()
            .any(|t| t.kind == task::TaskKind::Autolabel && t.state.is_busy())
            .then(|| "one is already queued".to_string())
    };
    let gate = std::sync::Barrier::new(8);
    let answers: Vec<std::result::Result<String, NotQueued>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let (svc, gate) = (&svc, &gate);
                s.spawn(move || {
                    gate.wait();
                    svc.enqueue_task_unless(
                        Box::new(HoldTask {
                            id: format!("t{i}"),
                            keep: false,
                        }),
                        &one_at_a_time,
                    )
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let queued: Vec<&String> = answers.iter().filter_map(|a| a.as_ref().ok()).collect();
    assert_eq!(queued.len(), 1, "{answers:?}");
    assert!(answers.iter().all(|a| match a {
        Ok(_) => true,
        Err(e) => *e == NotQueued::Refused("one is already queued".into()),
    }));
    assert_eq!(svc.tasks().len(), 1);
    // The plain enqueue has no rule; a repeated id is the queue's own failure.
    let again = svc.enqueue_task_unless(
        Box::new(HoldTask {
            id: queued[0].clone(),
            keep: false,
        }),
        &|_| None,
    );
    assert!(matches!(again, Err(NotQueued::Failed(m)) if m.contains("exists")));
    assert!(svc.cancel(queued[0]));
    svc.wait_task(queued[0], Duration::from_secs(10)).unwrap();
    svc.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A task that panics in its run (a bug in an import, an auto-label, a progress parser).
struct PanicTask {
    id: String,
}

impl Task for PanicTask {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> task::TaskKind {
        task::TaskKind::Import
    }
    fn title(&self) -> String {
        format!("panic {}", self.id)
    }
    fn steps(&self) -> usize {
        1
    }
    fn run(
        &self,
        _ctx: TaskContext<'_>,
        _report: &mut dyn FnMut(TaskReport<'_>),
        _cancel: &AtomicBool,
    ) -> Result<task::TaskOutcome> {
        panic!("index 812 out of range");
    }
}

/// A panic ends its job, never the queue: the record fails with the panic's message, `current`
/// is cleared (so the queue is not busy and the next job runs), and the queue's files say so.
#[test]
fn a_panicking_task_fails_and_the_queue_goes_on() {
    let dir = std::env::temp_dir().join(format!("bop-train-panic-{}", uuid::Uuid::new_v4()));
    let svc = TrainingService::start(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        Some(dir.join("runs.json")),
    );
    let ran = Arc::new(AtomicUsize::new(0));
    let bad = svc
        .enqueue_task(Box::new(PanicTask { id: "bad".into() }))
        .unwrap();
    let next = svc
        .enqueue_task(Box::new(QuickTask::new("next", None, &ran)))
        .unwrap();
    let r = svc.wait_task(&bad, Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Failed);
    let e = r.error.unwrap();
    assert!(
        e.starts_with("internal error: index 812 out of range."),
        "{e}"
    );
    assert!(e.contains("start the job again"), "{e}");
    assert!(r.finished.is_some());
    let r = svc.wait_task(&next, Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Done, "the queue kept running");
    assert_eq!(ran.load(Ordering::Relaxed), 1);
    assert!(svc.inner.current.lock_recover().is_none());
    assert!(!svc.busy());
    let back = task::load_records(&dir.join(task::TASKS_FILE));
    let saved = back.iter().find(|t| t.id == bad).unwrap();
    assert_eq!(saved.state, RunState::Failed);
    assert!(saved.error.as_deref().unwrap().contains("812"));
    // And still more jobs run after it.
    let again = svc
        .enqueue_task(Box::new(QuickTask::new("again", None, &ran)))
        .unwrap();
    assert_eq!(
        svc.wait_task(&again, Duration::from_secs(10))
            .unwrap()
            .state,
        RunState::Done
    );
    svc.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// The outer catch: a panic outside the job's own (here in a task's `after` rule) still fails
/// that record and leaves the queue running.
#[test]
fn a_panic_outside_the_job_still_fails_its_record() {
    struct PanicRule(QuickTask);
    impl Task for PanicRule {
        fn id(&self) -> &str {
            self.0.id()
        }
        fn kind(&self) -> task::TaskKind {
            self.0.kind()
        }
        fn title(&self) -> String {
            self.0.title()
        }
        fn steps(&self) -> usize {
            1
        }
        fn after(&self) -> Option<&str> {
            Some("first")
        }
        fn skip_reason(&self, _dep: &TaskRecord) -> Option<String> {
            panic!("the rule broke");
        }
        fn run(
            &self,
            ctx: TaskContext<'_>,
            report: &mut dyn FnMut(TaskReport<'_>),
            cancel: &AtomicBool,
        ) -> Result<task::TaskOutcome> {
            self.0.run(ctx, report, cancel)
        }
    }
    let svc = quick_service();
    let ran = Arc::new(AtomicUsize::new(0));
    svc.enqueue_task(Box::new(QuickTask::new("first", None, &ran)))
        .unwrap();
    let id = svc
        .enqueue_task(Box::new(PanicRule(QuickTask::new("second", None, &ran))))
        .unwrap();
    let r = svc.wait_task(&id, Duration::from_secs(10)).unwrap();
    assert_eq!(r.state, RunState::Failed);
    assert!(r.error.unwrap().contains("the rule broke"));
    let last = svc
        .enqueue_task(Box::new(QuickTask::new("third", None, &ran)))
        .unwrap();
    assert_eq!(
        svc.wait_task(&last, Duration::from_secs(10)).unwrap().state,
        RunState::Done
    );
    svc.shutdown();
}

/// A stopped queue takes no job and records none; a job whose message finds no thread fails
/// rather than staying queued (which would keep the queue busy until a restart).
#[test]
fn a_stopped_queue_refuses_jobs_and_a_lost_thread_leaves_no_queued_record() {
    let ran = Arc::new(AtomicUsize::new(0));
    let svc = quick_service();
    svc.shutdown();
    let e = svc
        .enqueue_task_unless(Box::new(QuickTask::new("late", None, &ran)), &|_| None)
        .unwrap_err();
    assert_eq!(e, NotQueued::Failed(STOPPED.into()));
    assert!(svc.tasks().is_empty(), "nothing recorded");
    assert!(!svc.busy());
    assert_eq!(svc.remove("x").unwrap_err(), STOPPED);
    assert_eq!(
        svc.set_store(Path::new("/elsewhere/runs.json"))
            .unwrap_err(),
        STOPPED
    );

    let dir = std::env::temp_dir().join(format!("bop-train-gone-{}", uuid::Uuid::new_v4()));
    let svc = TrainingService::start(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        Some(dir.join("runs.json")),
    );
    svc.end_thread_only();
    let e = svc
        .enqueue_task_unless(Box::new(QuickTask::new("lost", None, &ran)), &|_| None)
        .unwrap_err();
    assert_eq!(e, NotQueued::Failed(THREAD_GONE.into()));
    let r = svc.get_task("lost").unwrap();
    assert_eq!(r.state, RunState::Failed);
    assert_eq!(r.error.as_deref(), Some(THREAD_GONE));
    assert!(!svc.busy());
    let back = task::load_records(&dir.join(task::TASKS_FILE));
    assert_eq!(back[0].state, RunState::Failed);
    assert_eq!(back[0].error.as_deref(), Some(THREAD_GONE));
    assert_eq!(ran.load(Ordering::Relaxed), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The stop for exit: the running task's helper is stopped and the task, like the queued one,
/// is written as interrupted (the words a restart gives a crash's busy record), the thread ends,
/// and the queue never writes its files again.
#[test]
fn stop_for_exit_interrupts_the_running_and_queued_jobs() {
    let dir = std::env::temp_dir().join(format!("bop-train-exit-{}", uuid::Uuid::new_v4()));
    let runner = Arc::new(HoldRunner::default());
    let svc = TrainingService::start(
        runner.clone(),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        Some(dir.join("runs.json")),
    );
    let ran = Arc::new(AtomicUsize::new(0));
    let first = svc
        .enqueue_task(Box::new(HoldTask {
            id: "first".into(),
            keep: true,
        }))
        .unwrap();
    let second = svc
        .enqueue_task(Box::new(QuickTask::new("second", None, &ran)))
        .unwrap();
    until("the helper", || runner.started.load(Ordering::Relaxed) == 1);
    let t = Instant::now();
    assert!(svc.stop_for_exit(Instant::now() + Duration::from_secs(10)));
    assert!(t.elapsed() < Duration::from_secs(5));
    assert!(svc.thread.lock_recover().is_none(), "the thread was joined");
    assert!(svc.stopped());
    assert!(!svc.busy());
    let file = dir.join(task::TASKS_FILE);
    let crash_words = {
        let mut busy = svc.get_task(&first).unwrap();
        busy.state = RunState::Running;
        busy.error = None;
        let tmp = dir.join("crash.json");
        task::save_records(&tmp, &[busy]);
        task::load_records(&tmp).remove(0).error
    };
    for id in [&first, &second] {
        let r = svc.get_task(id).unwrap();
        assert_eq!(r.state, RunState::Failed, "{id}");
        assert!(r.finished.is_some());
        assert_eq!(
            r.error, crash_words,
            "{id}: the words of an interrupted job"
        );
        let saved = task::load_records(&file);
        let s = saved.iter().find(|s| s.id == *id).unwrap();
        assert_eq!(s.error, crash_words);
    }
    // What it kept, it keeps (a task that stops cleanly on the cancel).
    let r = svc.get_task(&first).unwrap();
    assert_eq!(r.summary.as_deref(), Some("kept 3 labelled frames"));
    assert_eq!(ran.load(Ordering::Relaxed), 0, "the queued job never ran");
    // No more writes: the next queue of the process owns the files.
    let before = std::fs::read_to_string(&file).unwrap();
    svc.inner
        .update_persist_task(&first, |r| r.detail = "late".into());
    assert_eq!(std::fs::read_to_string(&file).unwrap(), before);
    assert!(
        svc.enqueue_task(Box::new(QuickTask::new("late", None, &ran)))
            .is_err()
    );
    // Idempotent.
    assert!(svc.stop_for_exit(Instant::now()));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Past the deadline the thread is left behind (a job that ignores its cancel), its record is
/// written as interrupted at once, and a later Drop of the queue does not wait for it.
#[test]
fn stop_for_exit_gives_up_on_the_thread_at_its_deadline() {
    struct Stubborn {
        release: Arc<AtomicBool>,
    }
    impl Task for Stubborn {
        fn id(&self) -> &str {
            "stubborn"
        }
        fn kind(&self) -> task::TaskKind {
            task::TaskKind::Import
        }
        fn title(&self) -> String {
            "stubborn".into()
        }
        fn steps(&self) -> usize {
            1
        }
        fn run(
            &self,
            _ctx: TaskContext<'_>,
            _report: &mut dyn FnMut(TaskReport<'_>),
            _cancel: &AtomicBool,
        ) -> Result<task::TaskOutcome> {
            while !self.release.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(5));
            }
            anyhow::bail!("ended late")
        }
    }
    let dir = std::env::temp_dir().join(format!("bop-train-stuck-{}", uuid::Uuid::new_v4()));
    let release = Arc::new(AtomicBool::new(false));
    let svc = TrainingService::start(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        Some(dir.join("runs.json")),
    );
    let id = svc
        .enqueue_task(Box::new(Stubborn {
            release: release.clone(),
        }))
        .unwrap();
    until("the start", || state(&svc, &id) == RunState::Running);
    let t = Instant::now();
    assert!(!svc.stop_for_exit(Instant::now() + Duration::from_millis(100)));
    assert!(t.elapsed() < Duration::from_secs(5));
    let saved = task::load_records(&dir.join(task::TASKS_FILE));
    assert_eq!(saved[0].state, RunState::Failed);
    assert!(
        saved[0]
            .error
            .as_deref()
            .unwrap()
            .starts_with("interrupted")
    );
    let t = Instant::now();
    drop(svc);
    assert!(
        t.elapsed() < Duration::from_secs(1),
        "the drop did not join"
    );
    release.store(true, Ordering::Relaxed);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_closed_queue_refuses_everything() {
    let svc = TrainingService::closed(
        Arc::new(HoldRunner::default()),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
    );
    let ran = Arc::new(AtomicUsize::new(0));
    assert!(svc.stopped());
    assert!(svc.store().is_none());
    assert!(
        svc.enqueue_task(Box::new(QuickTask::new("a", None, &ran)))
            .is_err()
    );
    assert!(svc.tasks().is_empty() && svc.runs().is_empty());
    assert!(svc.stop_for_exit(Instant::now()));
}

#[test]
fn a_panic_message_is_read_from_either_payload() {
    let p = std::panic::catch_unwind(|| panic!("plain")).unwrap_err();
    assert_eq!(panic_text(p.as_ref()), "plain");
    let n = 3;
    let p = std::panic::catch_unwind(|| panic!("formatted {n}")).unwrap_err();
    assert_eq!(panic_text(p.as_ref()), "formatted 3");
    let p = std::panic::catch_unwind(|| std::panic::panic_any(7u8)).unwrap_err();
    assert_eq!(panic_text(p.as_ref()), "a panic without a message");
}

fn hold_service(runner: &Arc<HoldRunner>, store: Option<PathBuf>) -> Arc<TrainingService> {
    TrainingService::start(
        runner.clone(),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        store,
    )
}

fn lock_of(store: &Path) -> PathBuf {
    task::queue_lock_file(store)
}

/// QA-012 / RA-053: two queues on one folder (the service and a `train` command). The second is a
/// guest: it shows the owner's running job as running, refuses every job and removal with the
/// message naming the folder, cancels nothing, writes nothing, and the CLI's start is refused.
/// Once the owner is gone, the guest's next refresh makes it the owner.
#[test]
fn a_second_queue_on_a_folder_is_a_guest_until_the_owner_is_gone() {
    let dir = std::env::temp_dir().join(format!("bop-queue-guest-{}", uuid::Uuid::new_v4()));
    let store = dir.join(RUNS_FILE);
    let runner = Arc::new(HoldRunner::default());
    let owner = hold_service(&runner, Some(store.clone()));
    assert!(!owner.foreign_owner());
    assert!(lock_of(&store).is_file());
    let id = owner
        .enqueue_task(Box::new(HoldTask {
            id: "busy".into(),
            keep: false,
        }))
        .unwrap();
    until("the owner's job", || {
        runner.started.load(Ordering::Relaxed) == 1
    });
    until("its stage saved", || {
        task::read_records::<TaskRecord>(&task::tasks_file(&store))
            .first()
            .is_some_and(|r| r.state == RunState::Running && r.stage.is_some())
    });
    let tasks_file = task::tasks_file(&store);
    let before = (
        std::fs::read(&tasks_file).unwrap(),
        std::fs::read(&store).ok(),
    );
    let guest = hold_service(&runner, Some(store.clone()));
    assert!(guest.foreign_owner());
    assert_eq!(guest.store().as_deref(), Some(store.as_path()));
    assert_eq!(guest.get_task(&id).unwrap().state, RunState::Running);
    let ran = Arc::new(AtomicUsize::new(0));
    let refusal = guest_refusal(&dir);
    assert!(
        refusal.starts_with(
            "another blue-onyx-prism process (a `train` command) is using the training folder"
        ),
        "{refusal}"
    );
    assert_eq!(
        guest
            .enqueue_task(Box::new(QuickTask::new("q", None, &ran)))
            .unwrap_err(),
        refusal
    );
    assert_eq!(guest.remove(&id).unwrap_err(), refusal);
    assert_eq!(guest.while_idle(&|_, _| None, || ()).unwrap_err(), refusal);
    assert!(
        !guest.cancel(&id),
        "the owner's job is not the guest's to cancel"
    );
    // Still a guest after a refresh, which reads the owner's records again.
    guest.refresh();
    assert!(guest.foreign_owner());
    assert_eq!(guest.get_task(&id).unwrap().state, RunState::Running);
    // A CLI command refuses to run at all.
    let e = TrainingService::start_owner(
        runner.clone(),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        store.clone(),
    )
    .unwrap_err();
    assert_eq!(e, owner_refusal(&dir));
    assert!(e.starts_with("the service's training queue owns "), "{e}");
    assert!(
        e.ends_with("; queue it on the Train tab, or stop the service and run the command again"),
        "{e}"
    );
    assert_eq!(
        (
            std::fs::read(&tasks_file).unwrap(),
            std::fs::read(&store).ok()
        ),
        before,
        "nobody but the owner wrote"
    );
    assert_eq!(owner.get_task(&id).unwrap().state, RunState::Running);

    // The owner ends its job and goes: the guest owns the folder on its next refresh.
    assert!(owner.cancel(&id));
    owner.wait_task(&id, Duration::from_secs(10)).unwrap();
    owner.shutdown();
    assert!(
        !task::queue_owned(&lock_of(&store)),
        "a shut-down queue lets go"
    );
    guest.refresh();
    assert!(!guest.foreign_owner());
    assert!(task::queue_owned(&lock_of(&store)));
    assert_eq!(guest.get_task(&id).unwrap().state, RunState::Cancelled);
    let q = guest
        .enqueue_task(Box::new(QuickTask::new("q", None, &ran)))
        .unwrap();
    assert_eq!(
        guest.wait_task(&q, Duration::from_secs(10)).unwrap().state,
        RunState::Done
    );
    drop(guest);
    assert!(
        !task::queue_owned(&lock_of(&store)),
        "a dropped queue lets go"
    );
    assert!(lock_of(&store).is_file(), "the lock file is never deleted");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The guest never turns the owner's running record into an interrupted one, in memory or on
/// disk; the owner that opens a folder nobody holds does, and saves it.
#[test]
fn only_an_owner_interrupts_a_busy_record_and_saves_it() {
    let dir = std::env::temp_dir().join(format!("bop-queue-interrupt-{}", uuid::Uuid::new_v4()));
    let store = dir.join(RUNS_FILE);
    let tasks_file = task::tasks_file(&store);
    let mut rec = TaskRecord::new(
        &HoldTask {
            id: "elsewhere".into(),
            keep: false,
        },
        "c".into(),
    );
    rec.state = RunState::Running;
    task::save_records(&tasks_file, &[rec]);
    let saved = std::fs::read(&tasks_file).unwrap();
    let other = task::QueueLock::try_take(&lock_of(&store), Duration::ZERO)
        .unwrap()
        .unwrap();
    let runner = Arc::new(HoldRunner::default());
    let guest = hold_service(&runner, Some(store.clone()));
    assert!(guest.foreign_owner());
    assert_eq!(guest.tasks()[0].state, RunState::Running);
    guest.shutdown();
    assert_eq!(std::fs::read(&tasks_file).unwrap(), saved);
    drop(other);
    let owner = hold_service(&runner, Some(store.clone()));
    assert!(!owner.foreign_owner());
    assert_eq!(owner.tasks()[0].state, RunState::Failed);
    let on_disk = task::read_records::<TaskRecord>(&tasks_file);
    assert_eq!(on_disk[0].state, RunState::Failed, "saved at the open");
    assert!(
        on_disk[0]
            .error
            .as_deref()
            .unwrap()
            .starts_with("interrupted")
    );
    owner.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// `set_store` takes the new folder's lock before it lets the old one go, and refuses a folder
/// another queue owns, staying where it was.
#[test]
fn set_store_moves_the_folders_lock_and_refuses_an_owned_folder() {
    let dir = std::env::temp_dir().join(format!("bop-queue-move-{}", uuid::Uuid::new_v4()));
    let (a, b, c) = (
        dir.join("a").join(RUNS_FILE),
        dir.join("b").join(RUNS_FILE),
        dir.join("c").join(RUNS_FILE),
    );
    let runner = Arc::new(HoldRunner::default());
    let svc = hold_service(&runner, Some(a.clone()));
    assert!(task::queue_owned(&lock_of(&a)));
    assert!(svc.set_store(&b).unwrap());
    assert!(!task::queue_owned(&lock_of(&a)), "the old folder is free");
    assert!(task::queue_owned(&lock_of(&b)));
    let held = task::QueueLock::try_take(&lock_of(&c), Duration::ZERO)
        .unwrap()
        .unwrap();
    let e = svc.set_store(&c).unwrap_err();
    assert_eq!(e, guest_refusal(&dir.join("c")));
    assert_eq!(svc.store().as_deref(), Some(b.as_path()), "it stays");
    assert!(!svc.foreign_owner());
    assert!(task::queue_owned(&lock_of(&b)), "and keeps its lock");
    drop(held);
    assert!(svc.set_store(&c).unwrap());
    assert!(!task::queue_owned(&lock_of(&b)));
    svc.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A stop for exit that leaves its thread behind still lets the folder go: the next queue of
/// the process (an in-process restart) owns it.
#[test]
fn stop_for_exit_lets_the_folder_go_even_with_its_thread_left_behind() {
    struct Stuck(Arc<AtomicBool>);
    impl Task for Stuck {
        fn id(&self) -> &str {
            "stuck"
        }
        fn kind(&self) -> task::TaskKind {
            task::TaskKind::Import
        }
        fn title(&self) -> String {
            "stuck".into()
        }
        fn steps(&self) -> usize {
            1
        }
        fn run(
            &self,
            _ctx: TaskContext<'_>,
            _report: &mut dyn FnMut(TaskReport<'_>),
            _cancel: &AtomicBool,
        ) -> Result<task::TaskOutcome> {
            while !self.0.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(5));
            }
            anyhow::bail!("ended late")
        }
    }
    let dir = std::env::temp_dir().join(format!("bop-queue-exit-lock-{}", uuid::Uuid::new_v4()));
    let store = dir.join(RUNS_FILE);
    let release = Arc::new(AtomicBool::new(false));
    let runner = Arc::new(HoldRunner::default());
    let old = hold_service(&runner, Some(store.clone()));
    let id = old.enqueue_task(Box::new(Stuck(release.clone()))).unwrap();
    until("the start", || state(&old, &id) == RunState::Running);
    assert!(!old.stop_for_exit(Instant::now() + Duration::from_millis(50)));
    let new = hold_service(&runner, Some(store.clone()));
    assert!(!new.foreign_owner(), "the next queue owns the folder");
    new.shutdown();
    release.store(true, Ordering::Relaxed);
    drop(old);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The other process of [`a_queue_of_another_process_makes_this_one_a_guest`]; idle in a normal
/// run. Holds a queue on the folder until the release file appears.
#[test]
fn queue_owner_child() {
    let Ok(dir) = std::env::var("BOP_TEST_QUEUE_OWNER_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let svc = hold_service(&Arc::new(HoldRunner::default()), Some(dir.join(RUNS_FILE)));
    assert!(!svc.foreign_owner());
    std::fs::write(dir.join("ready"), b"").unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !dir.join("release").exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    svc.shutdown();
}

/// Two processes: while the other process's queue owns the folder, a queue here is a guest and
/// the CLI's start is refused with its message; when that process ends, the guest becomes the
/// owner.
#[test]
fn a_queue_of_another_process_makes_this_one_a_guest() {
    let dir = std::env::temp_dir().join(format!("bop-queue-2proc-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = dir.join(RUNS_FILE);
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["training::service::tests::queue_owner_child", "--exact"])
        .env("BOP_TEST_QUEUE_OWNER_DIR", &dir)
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    until("the other process's queue", || dir.join("ready").exists());
    let runner = Arc::new(HoldRunner::default());
    let guest = hold_service(&runner, Some(store.clone()));
    assert!(guest.foreign_owner());
    let e = TrainingService::start_owner(
        runner.clone(),
        Arc::new(NoFetch),
        Arc::new(super::super::job::NoHooks),
        store.clone(),
    )
    .unwrap_err();
    assert_eq!(e, owner_refusal(&dir));
    std::fs::write(dir.join("release"), b"").unwrap();
    assert!(child.wait().unwrap().success());
    guest.refresh();
    assert!(!guest.foreign_owner());
    guest.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}

/// `while_idle`: an enqueue from another thread waits until the action returns; a busy queue is
/// refused (the action never runs); the caller's rule refuses with its own words.
#[test]
fn while_idle_holds_every_enqueue_and_refuses_a_busy_queue() {
    let runner = Arc::new(HoldRunner::default());
    let svc = hold_service(&runner, None);
    let ran = Arc::new(AtomicUsize::new(0));
    let acting = Arc::new(AtomicBool::new(false));
    let acted = Arc::new(AtomicBool::new(false));
    std::thread::scope(|s| {
        let (svc2, acting2, acted2) = (&svc, acting.clone(), acted.clone());
        let act = s.spawn(move || {
            svc2.while_idle(&|_, _| None, || {
                acting2.store(true, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(200));
                acted2.store(true, Ordering::SeqCst);
            })
        });
        until("the action", || acting.load(Ordering::SeqCst));
        svc.enqueue_task(Box::new(QuickTask::new("after", None, &ran)))
            .unwrap();
        assert!(
            acted.load(Ordering::SeqCst),
            "the enqueue waited for the action"
        );
        act.join().unwrap().unwrap();
    });
    svc.wait_task("after", Duration::from_secs(10)).unwrap();
    assert_eq!(
        svc.while_idle(&|_, _| Some("not now".into()), || unreachable!())
            .unwrap_err(),
        "not now"
    );
    let id = svc
        .enqueue_task(Box::new(HoldTask {
            id: "hold".into(),
            keep: false,
        }))
        .unwrap();
    let e = svc.while_idle(&|_, _| None, || unreachable!()).unwrap_err();
    assert!(e.contains("queued or running"), "{e}");
    assert!(svc.cancel(&id));
    svc.wait_task(&id, Duration::from_secs(10)).unwrap();
    svc.shutdown();
}

/// quality extra 43: planning a dataset check reads `runs.json` as a reader: a damaged one stays
/// where it is (the queue's owner deals with it).
#[test]
fn planning_a_dataset_check_never_moves_runs_json_aside() {
    let root = std::env::temp_dir().join(format!("bop-plan-refine-{}", uuid::Uuid::new_v4()));
    let mut config = crate::config::Config {
        download_dir: Some(root.clone()),
        allow_large_downloads: true,
        ..crate::config::Config::default()
    };
    config.seed_default_model();
    config.collect.dir = Some(root.join("dataset"));
    let store = super::super::runs_file(&config);
    std::fs::create_dir_all(store.parent().unwrap()).unwrap();
    std::fs::write(&store, b"[{\"id\": \"cut off").unwrap();
    super::super::refine::plan_refine(
        &config,
        super::super::refine::RefineRequest::default(),
        &super::super::capability::host_for_tests(),
    )
    .unwrap();
    assert_eq!(std::fs::read(&store).unwrap(), b"[{\"id\": \"cut off");
    let names: Vec<String> = std::fs::read_dir(store.parent().unwrap())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(!names.iter().any(|n| n.contains(".corrupt-")), "{names:?}");
    let _ = std::fs::remove_dir_all(&root);
}

/// QA-039: `--verbose` stopped after 60 lines (the CLI followed the tail by its length, which
/// stops at the cap). 200 lines go through [`push_log`]; the follower reads after varying
/// batches, one of them larger than the tail: every line it returns is returned once, in order,
/// and only the lines evicted before that read are missing.
#[test]
fn the_log_follower_returns_each_line_once_past_the_tail() {
    let mut log = Vec::new();
    let mut seq = 0u64;
    let mut follow = LogFollower::new();
    let mut got: Vec<String> = Vec::new();
    let mut pushed = 0usize;
    // Batches between two reads: small ones, one of 90 (> LOG_TAIL), then small ones again.
    for batch in [1usize, 5, 30, 60, 90, 3, 7, 4]
        .into_iter()
        .chain(std::iter::repeat(1))
    {
        if pushed >= 200 {
            break;
        }
        for _ in 0..batch.min(200 - pushed) {
            push_log(&mut log, &mut seq, &format!("line {pushed}"));
            pushed += 1;
        }
        assert!(log.len() <= LOG_TAIL);
        got.extend(follow.new_lines(&log, seq).iter().cloned());
        // A read with nothing new returns nothing.
        assert!(follow.new_lines(&log, seq).is_empty());
    }
    assert_eq!(seq, 200);
    let numbers: Vec<usize> = got
        .iter()
        .map(|l| l.strip_prefix("line ").unwrap().parse().unwrap())
        .collect();
    assert!(numbers.windows(2).all(|w| w[0] < w[1]), "once, in order");
    // The 90-line batch (lines 96 to 185) lost its first 30, evicted before the read.
    let expected: Vec<usize> = (0..96).chain(126..200).collect();
    assert_eq!(numbers, expected);
    // Long lines are cut, and a tail trimmed after the fact (a finished run) still follows.
    push_log(&mut log, &mut seq, &"x".repeat(1000));
    assert_eq!(log.last().unwrap().chars().count(), LOG_LINE_CHARS);
    let skip = log.len() - 15;
    log.drain(..skip);
    assert_eq!(follow.new_lines(&log, seq).len(), 1);
}

/// A busy run's and a busy task's line: the stage, its step and the detail, "paused (...)" while
/// paused, "starting" / "paused" before the first stage ([`progress_text`], shared by both
/// `describe`s).
#[test]
fn busy_runs_and_tasks_describe_their_stage_alike() {
    let recipe =
        crate::training::TrainRecipe::new(crate::resources::catalog::TrainFamily::Yolo26, "n")
            .unwrap();
    let mut r: RunRecord = serde_json::from_value(serde_json::json!({
        "id": "r1", "name": "my-model", "created": "",
        "request": {"recipe": recipe, "datasets": []},
        "device": "mps", "state": "running", "stage": "train", "step": 5, "steps": 8,
        "percent": 40, "detail": "epoch 3/50", "run_dir": "runs/r1"
    }))
    .unwrap();
    let train = TrainStage::Train.label();
    assert_eq!(r.describe(), format!("{train} (5/8): epoch 3/50"));
    r.state = RunState::Paused;
    assert_eq!(r.describe(), format!("paused ({train} 5/8): epoch 3/50"));
    r.detail.clear();
    assert_eq!(r.describe(), format!("paused ({train} 5/8)"));
    r.state = RunState::Running;
    assert_eq!(r.describe(), format!("{train} (5/8)"));
    r.stage = None;
    assert_eq!(r.describe(), "starting");
    r.state = RunState::Paused;
    assert_eq!(r.describe(), "paused");

    // A task's arms say the same of the same stage.
    let mut t = task::TaskRecord {
        id: "t1".into(),
        kind: task::TaskKind::Import,
        title: "import".into(),
        created: String::new(),
        state: RunState::Running,
        stage: Some("extracting frames".into()),
        step: 2,
        steps: 5,
        percent: 30,
        detail: String::new(),
        log: vec![],
        summary: None,
        result: None,
        error: None,
        finished: None,
        after: None,
        uses: Vec::new(),
    };
    for paused in [false, true] {
        t.state = if paused {
            RunState::Paused
        } else {
            RunState::Running
        };
        for detail in ["", "12 frames"] {
            t.detail = detail.into();
            t.stage = Some("extracting frames".into());
            assert_eq!(
                t.describe(),
                progress_text(paused, Some("extracting frames"), 2, 5, detail)
            );
            t.stage = None;
            assert_eq!(t.describe(), progress_text(paused, None, 2, 5, detail));
        }
    }
    assert_eq!(
        progress_text(false, Some("labelling"), 2, 2, "3 of 6 frames"),
        "labelling (2/2): 3 of 6 frames"
    );
}

/// A stage's detail keeps its first 300 characters (characters, not bytes).
#[test]
fn a_detail_keeps_its_first_300_characters() {
    let long = "é".repeat(400);
    assert_eq!(detail_text(&long).chars().count(), DETAIL_CHARS);
    assert_eq!(DETAIL_CHARS, 300);
    assert_eq!(detail_text("12 frames"), "12 frames");
}
