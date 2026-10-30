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
    until("the helper to see the pause", || {
        runner.paused.load(Ordering::Relaxed)
    });
    // Persisted as paused (a restart turns it into the interrupted failure).
    let raw = std::fs::read_to_string(dir.join(task::TASKS_FILE)).unwrap();
    assert!(raw.contains(r#""state": "paused""#), "{raw}");
    let back = task::load_records(&dir.join(task::TASKS_FILE));
    assert_eq!(back[0].state, RunState::Failed);
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
        valid_images: 40,
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
}
