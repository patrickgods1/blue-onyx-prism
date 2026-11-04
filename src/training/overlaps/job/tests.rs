//! The overlap check over fakes: the export's [`Fake`] for uv and downloads, a fake
//! `autolabel.py --zoom-pairs` that answers each crop, the conventions of the auto-label job's tests.

use super::*;
use crate::collect::labels::bulk::test_support::put_frame;
use crate::collect::labels::overlaps::Verdict;
use crate::collect::labels::{self, BoxState, Frame, LabelBox, LabelFile, Status};
use crate::config::Config;
use crate::resources::catalog::Resource;
use crate::resources::export::fake::{Fake, arg_after};
use crate::resources::export::{Cmd, CommandRunner, Fetcher};
use crate::training::capability::{self, Capability};
use crate::training::overlaps::{OverlapsRequest, plan_overlaps};
use std::path::Path;
use std::sync::Mutex;

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tmp(tag: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("bop-overlaps-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

const BIG: [f32; 4] = [1.0, 1.0, 41.0, 41.0];
const SMALL: [f32; 4] = [5.0, 5.0, 25.0, 25.0];
const FAR: [f32; 4] = [45.0, 30.0, 60.0, 46.0];

#[derive(Default)]
struct ZoomFake {
    inner: Fake,
    cmds: Mutex<Vec<Cmd>>,
    /// Detections of every crop (default: one car on [`BIG`]).
    dets: Option<serde_json::Value>,
    /// Cancel after writing this many lines, the next one cut off mid-line.
    stop_after: Option<usize>,
    /// Fail after writing this many lines.
    fail_after: Option<usize>,
    /// Runs of the script (`--zoom-pairs`).
    zooms: Mutex<usize>,
    /// Called once the script started: a reviewer working during the run.
    meanwhile: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl CommandRunner for ZoomFake {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        if !cmd.display().contains("autolabel.py") {
            return self.inner.run(cmd, on_line, cancel);
        }
        if cmd.args.iter().any(|a| a == "--self-check") {
            on_line("@autolabel {\"event\":\"self_test\",\"passed\":68}");
            return Ok(());
        }
        self.cmds.lock().unwrap().push(cmd.clone());
        *self.zooms.lock().unwrap() += 1;
        if let Some(f) = self.meanwhile.lock().unwrap().take() {
            f();
        }
        let list = arg_after(cmd, "--zoom-pairs").unwrap();
        let out = arg_after(cmd, "--out").unwrap();
        let dets = self.dets.clone().unwrap_or_else(
            || serde_json::json!([{"label": "car", "bbox": [1, 1, 41, 41], "score": 0.82}]),
        );
        let mut text = String::new();
        let mut n = 0;
        for l in std::fs::read_to_string(list).unwrap().lines() {
            let f: serde_json::Value = serde_json::from_str(l).unwrap();
            for c in f["crops"].as_array().unwrap() {
                if self.stop_after == Some(n) {
                    text.push_str("{\"id\":99,\"fr");
                    std::fs::write(&out, &text).unwrap();
                    cancel.store(true, Ordering::SeqCst);
                    return Err(anyhow!(crate::resources::export::Cancelled));
                }
                if self.fail_after == Some(n) {
                    std::fs::write(&out, &text).unwrap();
                    bail!("the script died");
                }
                let line = serde_json::json!({"id": c["id"], "frame": f["frame"], "boxes": dets});
                text.push_str(&line.to_string());
                text.push('\n');
                n += 1;
                on_line(&format!("@overlaps {{\"event\":\"crop\",\"index\":{n}}}"));
            }
        }
        std::fs::write(out, text).unwrap();
        on_line("@overlaps {\"event\":\"done\"}");
        Ok(())
    }
}

impl Fetcher for ZoomFake {
    fn fetch(
        &self,
        res: &'static Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.inner.fetch(res, dir, progress, cancel)
    }
}

fn config(root: &Path) -> Config {
    let mut c = Config {
        download_dir: Some(root.to_path_buf()),
        allow_large_downloads: true,
        ..Config::default()
    };
    c.seed_default_model();
    c.collect.dir = Some(root.join("dataset"));
    c
}

fn host() -> Capability {
    capability::host_for_tests()
}

fn plan(root: &Path, filter: &str) -> OverlapsJob {
    let cfg = config(root);
    let mut job = plan_overlaps(&cfg, OverlapsRequest::parse(filter).unwrap(), &host()).unwrap();
    job.free_commit = || None;
    job
}

fn run(job: &OverlapsJob, fake: &ZoomFake) -> Result<TaskOutcome> {
    run_cancellable(job, fake, &AtomicBool::new(false))
}

fn run_cancellable(job: &OverlapsJob, fake: &ZoomFake, cancel: &AtomicBool) -> Result<TaskOutcome> {
    job.run(
        TaskContext {
            runner: fake,
            fetcher: fake,
        },
        &mut |_| {},
        cancel,
    )
}

fn labelled(label: &str, bbox: [f32; 4], labelers: &[&str]) -> LabelBox {
    let mut b = crate::collect::labels::bulk::test_support::auto_box(
        label,
        0.5,
        crate::collect::labels::Triage::Review,
    );
    b.bbox = bbox;
    for l in labelers {
        b.auto.labelers.insert((*l).to_string(), 0.5);
    }
    b
}

fn nested(dir: &Path, stem: &str) -> String {
    put_frame(
        dir,
        stem,
        None,
        vec![
            labelled("car", BIG, &["gdino", "rfdetr"]),
            labelled("car", SMALL, &["gdino"]),
        ],
    )
}

fn path_of(dir: &Path, rel: &str) -> PathBuf {
    Frame::resolve(dir, rel).unwrap().labels
}

fn labels_of(dir: &Path, rel: &str) -> LabelFile {
    labels::read(&path_of(dir, rel)).unwrap().unwrap()
}

fn edit(dir: &Path, rel: &str, f: impl FnOnce(&mut LabelFile)) {
    let mut l = labels_of(dir, rel);
    f(&mut l);
    labels::write(&path_of(dir, rel), &l).unwrap();
}

#[test]
fn it_writes_suggestions_on_reviewed_and_unreviewed_frames_and_nothing_else() {
    let t = tmp("write");
    let dir = t.0.join("dataset");
    let unreviewed = nested(&dir, "u");
    let reviewed = nested(&dir, "r");
    let at = "2026-10-22T00:00:00Z";
    edit(&dir, &reviewed, |l| {
        l.status = Status::Reviewed;
        l.reviewed_at = Some(at.into());
        l.boxes[0].state = BoxState::Accepted;
        l.boxes[1].state = BoxState::Accepted;
    });
    let plain = put_frame(&dir, "plain", None, vec![labelled("car", FAR, &[])]);
    let before: Vec<Vec<u8>> = [&unreviewed, &reviewed, &plain]
        .iter()
        .map(|r| std::fs::read(path_of(&dir, r)).unwrap())
        .collect();
    let fake = ZoomFake::default();
    let job = plan(&t.0, "");
    assert_eq!(job.steps(), 4);
    assert_eq!(job.kind(), TaskKind::Overlaps);
    let out = run(&job, &fake).unwrap();
    assert!(
        out.summary.contains("No box was changed"),
        "{}",
        out.summary
    );
    assert_eq!(out.result["frames"], 2, "{}", out.result);
    assert_eq!(out.result["drops"], 2);
    assert_eq!(out.result["crops"], 2);
    assert_eq!(out.result["crops_done"], 2);
    // Byte-identical apart from the new field: take it out and the file is what it was.
    for (rel, was) in [&unreviewed, &reviewed].iter().zip(&before) {
        let mut l = labels_of(&dir, rel);
        assert_eq!(l.overlap_suggestions.len(), 1, "{rel}");
        assert_eq!(l.overlap_suggestions[0].verdict, Verdict::DropB);
        l.overlap_suggestions.clear();
        let mut bytes = serde_json::to_vec_pretty(&l).unwrap();
        bytes.push(b'\n');
        assert_eq!(&bytes, was, "{rel}");
    }
    let l = labels_of(&dir, &reviewed);
    assert_eq!(
        (l.status, l.reviewed_at.as_deref()),
        (Status::Reviewed, Some(at))
    );
    assert!(!l.overlaps_checked);
    // A frame with no pair is not written at all, nor is the review queue touched.
    assert_eq!(std::fs::read(path_of(&dir, &plain)).unwrap(), before[2]);
    // The work folder is gone.
    assert!(!job.work.exists());
    // What the script was told.
    let cmds = fake.cmds.lock().unwrap();
    let cmd = &cmds[0];
    let arg = |flag: &str| arg_after(cmd, flag).unwrap().to_string_lossy().into_owned();
    assert_eq!(arg("--rfdetr-threshold"), "0.1");
    assert_eq!(arg("--zoom-pad"), "0.2");
    let weights = arg_after(cmd, "--rfdetr").unwrap();
    assert!(
        weights.to_string_lossy().contains("rfdetr-large"),
        "{weights:?}"
    );
    assert!(cmd.display().contains("--device"));
    // RF-DETR L's weights came through the pinned download.
    let fetched = fake.inner.fetched.lock().unwrap();
    assert!(
        fetched.iter().any(|f| f == "train-base:rfdetr-large"),
        "{fetched:?}"
    );
}

#[test]
fn a_rerun_has_nothing_left_and_again_looks_afresh() {
    let t = tmp("rerun");
    let dir = t.0.join("dataset");
    let rel = nested(&dir, "a");
    let fake = ZoomFake::default();
    run(&plan(&t.0, ""), &fake).unwrap();
    let bytes = std::fs::read(path_of(&dir, &rel)).unwrap();
    let out = run(&plan(&t.0, ""), &fake).unwrap();
    assert!(
        out.summary.starts_with("nothing to check"),
        "{}",
        out.summary
    );
    assert_eq!(*fake.zooms.lock().unwrap(), 1, "no second model run");
    assert_eq!(std::fs::read(path_of(&dir, &rel)).unwrap(), bytes);
    // An unclear pair counts as answered too.
    let t2 = tmp("unclear");
    let dir2 = t2.0.join("dataset");
    let rel2 = put_frame(
        &dir2,
        "t",
        None,
        vec![
            labelled("car", BIG, &["gdino"]),
            labelled("car", SMALL, &["rfdetr"]),
        ],
    );
    let none = ZoomFake {
        dets: Some(serde_json::json!([])),
        ..ZoomFake::default()
    };
    let out = run(&plan(&t2.0, ""), &none).unwrap();
    assert_eq!(out.result["unclear"], 1);
    assert_eq!(
        labels_of(&dir2, &rel2).overlap_suggestions[0].verdict,
        Verdict::Unclear
    );
    assert!(
        run(&plan(&t2.0, ""), &none)
            .unwrap()
            .summary
            .starts_with("nothing")
    );
    // `again=1`: the same answer rewrites nothing; a new answer replaces the old one.
    let out = run(&plan(&t.0, "again=1"), &fake).unwrap();
    assert_eq!(out.result["crops"], 1);
    assert_eq!(out.result["frames"], 0, "{}", out.result);
    assert_eq!(std::fs::read(path_of(&dir, &rel)).unwrap(), bytes);
    let both = ZoomFake {
        dets: Some(serde_json::json!([
            {"label": "car", "bbox": [1, 1, 41, 41], "score": 0.9},
            {"label": "car", "bbox": [5, 5, 25, 25], "score": 0.8}
        ])),
        ..ZoomFake::default()
    };
    run(&plan(&t.0, "again=1"), &both).unwrap();
    let l = labels_of(&dir, &rel);
    assert_eq!(l.overlap_suggestions.len(), 1);
    assert_eq!(l.overlap_suggestions[0].verdict, Verdict::TwoObjects);
}

#[test]
fn a_cancel_keeps_the_finished_crops_and_a_rerun_resumes() {
    let t = tmp("cancel");
    let dir = t.0.join("dataset");
    let rels: Vec<String> = (0..3).map(|i| nested(&dir, &format!("f{i}"))).collect();
    let fake = ZoomFake {
        stop_after: Some(1),
        ..ZoomFake::default()
    };
    let cancel = AtomicBool::new(false);
    let out = run_cancellable(&plan(&t.0, ""), &fake, &cancel).unwrap();
    assert!(
        out.summary.starts_with("stopped by Cancel after 1 of 3"),
        "{}",
        out.summary
    );
    assert_eq!(out.result["cancelled"], true);
    assert_eq!(out.result["crops_done"], 1);
    let done: Vec<usize> = rels
        .iter()
        .map(|r| labels_of(&dir, r).overlap_suggestions.len())
        .collect();
    assert_eq!(done.iter().sum::<usize>(), 1, "{done:?}");
    // The rerun only asks for what is left.
    let again = ZoomFake::default();
    let job = plan(&t.0, "");
    assert_eq!(job.select().unwrap().pairs, 2);
    let out = run(&job, &again).unwrap();
    assert_eq!(out.result["crops"], 2);
    assert!(
        rels.iter()
            .all(|r| labels_of(&dir, r).overlap_suggestions.len() == 1)
    );
    // A cancel before anything finished is a plain cancel, and writes nothing.
    let t = tmp("cancel-early");
    let dir = t.0.join("dataset");
    let rel = nested(&dir, "a");
    let bytes = std::fs::read(path_of(&dir, &rel)).unwrap();
    let fake = ZoomFake {
        stop_after: Some(0),
        ..ZoomFake::default()
    };
    let e = run_cancellable(&plan(&t.0, ""), &fake, &AtomicBool::new(false)).unwrap_err();
    assert!(is_cancel(&e), "{e:#}");
    assert_eq!(std::fs::read(path_of(&dir, &rel)).unwrap(), bytes);
}

#[test]
fn a_failure_keeps_the_finished_crops_and_says_so() {
    let t = tmp("fail");
    let dir = t.0.join("dataset");
    let rels: Vec<String> = (0..2).map(|i| nested(&dir, &format!("f{i}"))).collect();
    let fake = ZoomFake {
        fail_after: Some(1),
        ..ZoomFake::default()
    };
    let e = run(&plan(&t.0, ""), &fake).unwrap_err();
    let text = format!("{e:#}");
    assert!(
        text.contains("finished crops were saved") && text.contains("the script died"),
        "{text}"
    );
    let n: usize = rels
        .iter()
        .map(|r| labels_of(&dir, r).overlap_suggestions.len())
        .sum();
    assert_eq!(n, 1);
    // Nothing finished: only the failure.
    let fake = ZoomFake {
        fail_after: Some(0),
        ..ZoomFake::default()
    };
    let e = run(&plan(&t.0, "again=1"), &fake).unwrap_err();
    assert!(!format!("{e:#}").contains("were saved"), "{e:#}");
}

#[test]
fn a_reviewer_working_during_the_run_wins() {
    let t = tmp("live");
    let dir = t.0.join("dataset");
    let moved = nested(&dir, "moved");
    let settled = nested(&dir, "settled");
    let checked = nested(&dir, "checked");
    let skipped = nested(&dir, "skipped");
    let kept = nested(&dir, "kept");
    let d = dir.clone();
    let (a, b, c, e) = (
        moved.clone(),
        settled.clone(),
        checked.clone(),
        skipped.clone(),
    );
    let fake = ZoomFake {
        meanwhile: Mutex::new(Some(Box::new(move || {
            edit(&d, &a, |l| l.boxes[1].bbox = [6.0, 5.0, 25.0, 25.0]);
            edit(&d, &b, |l| l.boxes[0].state = BoxState::Rejected);
            edit(&d, &c, |l| l.overlaps_checked = true);
            edit(&d, &e, |l| l.status = Status::Skipped);
        }))),
        ..ZoomFake::default()
    };
    let out = run(&plan(&t.0, ""), &fake).unwrap();
    assert_eq!(out.result["frames"], 1, "{}", out.result);
    assert_eq!(out.result["crops"], 5);
    for rel in [&moved, &settled, &checked, &skipped] {
        assert!(labels_of(&dir, rel).overlap_suggestions.is_empty(), "{rel}");
    }
    assert_eq!(labels_of(&dir, &kept).overlap_suggestions.len(), 1);
    // The reviewer's edits are all still there.
    assert_eq!(
        labels_of(&dir, &moved).boxes[1].bbox,
        [6.0, 5.0, 25.0, 25.0]
    );
    assert_eq!(labels_of(&dir, &skipped).status, Status::Skipped);
    // A frame deleted during the run is a warning, not a failed job.
    let t = tmp("gone");
    let dir = t.0.join("dataset");
    let gone = nested(&dir, "gone");
    let stay = nested(&dir, "stay");
    let img = Frame::resolve(&dir, &gone).unwrap().jpg;
    let fake = ZoomFake {
        meanwhile: Mutex::new(Some(Box::new(move || std::fs::remove_file(img).unwrap()))),
        ..ZoomFake::default()
    };
    let out = run(&plan(&t.0, ""), &fake).unwrap();
    assert_eq!(
        out.result["warnings"].as_array().unwrap().len(),
        1,
        "{}",
        out.result
    );
    assert_eq!(labels_of(&dir, &stay).overlap_suggestions.len(), 1);
}

#[test]
fn skipped_and_checked_frames_are_not_looked_at_and_the_limit_counts_work() {
    let t = tmp("scope");
    let dir = t.0.join("dataset");
    let skipped = nested(&dir, "a-skipped");
    let checked = nested(&dir, "b-checked");
    nested(&dir, "c");
    nested(&dir, "d");
    edit(&dir, &skipped, |l| l.status = Status::Skipped);
    edit(&dir, &checked, |l| l.overlaps_checked = true);
    let s = plan(&t.0, "").select().unwrap();
    assert_eq!((s.frame_count, s.pairs, s.passed_over), (2, 2, 2));
    let s = plan(&t.0, "limit=1").select().unwrap();
    assert_eq!(s.frame_count, 1);
    let s = plan(&t.0, "camera=elsewhere").select().unwrap();
    assert_eq!(s.frame_count, 0);
}

#[test]
fn hidden_classes_are_not_checked_and_a_class_rf_detr_cannot_see_needs_no_model() {
    let t = tmp("classes");
    let dir = t.0.join("dataset");
    let rel = put_frame(
        &dir,
        "r",
        None,
        vec![
            labelled("raccoon", BIG, &["gdino", "rfdetr"]),
            labelled("raccoon", SMALL, &["gdino"]),
        ],
    );
    let fake = ZoomFake::default();
    let job = plan(&t.0, "");
    let s = job.select().unwrap();
    assert_eq!((s.pairs, s.no_zoom, s.crops), (1, 1, 0));
    let out = run(&job, &fake).unwrap();
    assert_eq!(*fake.zooms.lock().unwrap(), 0, "no model for it");
    assert_eq!(out.result["low_confidence"], 1, "{}", out.result);
    let l = labels_of(&dir, &rel);
    assert_eq!(l.overlap_suggestions[0].verdict, Verdict::DropB);
    assert!(l.overlap_suggestions[0].low_confidence);
    assert!(fake.inner.fetched.lock().unwrap().is_empty(), "no download");
    // Hidden: not a pair at all.
    let mut job = plan(&t.0, "again=1");
    job.hidden = vec!["raccoon".into()];
    assert_eq!(job.select().unwrap().pairs, 0);
}

#[test]
fn short_commit_stops_with_a_clear_error_before_loading_anything() {
    let t = tmp("commit");
    let dir = t.0.join("dataset");
    let rel = nested(&dir, "a");
    let bytes = std::fs::read(path_of(&dir, &rel)).unwrap();
    let fake = ZoomFake::default();
    let mut job = plan(&t.0, "");
    job.free_commit = || {
        Some(Commit {
            limit: 16 << 30,
            available: 1 << 30,
        })
    };
    let e = run(&job, &fake).unwrap_err().to_string();
    assert!(
        e.contains("2.6 GB of free commit") && e.contains("1.0 GB is free"),
        "{e}"
    );
    assert_eq!(*fake.zooms.lock().unwrap(), 0);
    assert_eq!(std::fs::read(path_of(&dir, &rel)).unwrap(), bytes);
    let ok = Commit {
        limit: 16 << 30,
        available: 3 << 30,
    };
    assert!(commit_problem(Some(ok)).is_none());
    assert!(
        commit_problem(None).is_none(),
        "off Windows there is no fixed limit"
    );
}

#[test]
fn an_unreadable_frame_falls_back_to_the_witnesses() {
    let t = tmp("unreadable");
    let dir = t.0.join("dataset");
    let rel = nested(&dir, "a");
    #[derive(Default)]
    struct Broken(ZoomFake);
    impl CommandRunner for Broken {
        fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
            if !cmd.args.iter().any(|a| a == "--zoom-pairs") {
                return self.0.run(cmd, on_line, cancel);
            }
            std::fs::write(
                arg_after(cmd, "--out").unwrap(),
                "{\"id\":0,\"frame\":\"x\",\"error\":\"broken\"}\n",
            )?;
            on_line("@overlaps {\"event\":\"done\"}");
            Ok(())
        }
    }
    let fake = Broken::default();
    let job = plan(&t.0, "");
    job.run(
        TaskContext {
            runner: &fake,
            fetcher: &fake.0,
        },
        &mut |_| {},
        &AtomicBool::new(false),
    )
    .unwrap();
    let s = &labels_of(&dir, &rel).overlap_suggestions[0];
    assert_eq!((s.verdict, s.low_confidence), (Verdict::DropB, true));
}

#[test]
fn planning_needs_a_gpu_and_a_download_allowed() {
    let t = tmp("plan");
    let cfg = config(&t.0);
    let mut none = host();
    none.accelerator = None;
    let e = plan_overlaps(&cfg, OverlapsRequest::parse("").unwrap(), &none)
        .unwrap_err()
        .to_string();
    assert!(e.contains("NVIDIA GPU or Apple silicon"), "{e}");
    let closed = Config {
        allow_large_downloads: false,
        ..cfg
    };
    let e = plan_overlaps(&closed, OverlapsRequest::parse("").unwrap(), &host())
        .unwrap_err()
        .to_string();
    assert!(e.contains("--allow-large"), "{e}");
}

#[test]
fn the_queue_runs_it_like_any_other_job() {
    use crate::training::service::{RunState, TrainingService};
    let t = tmp("queue");
    let dir = t.0.join("dataset");
    let rel = nested(&dir, "a");
    let fake = std::sync::Arc::new(ZoomFake::default());
    let svc = TrainingService::start(
        fake.clone(),
        fake.clone(),
        std::sync::Arc::new(crate::training::job::NoHooks),
        Some(t.0.join("runs.json")),
    );
    let id = svc.enqueue_task(Box::new(plan(&t.0, ""))).unwrap();
    let r = svc.wait_task(&id, Duration::from_secs(60)).unwrap();
    assert_eq!(r.state, RunState::Done, "{r:?}");
    assert_eq!(r.kind, TaskKind::Overlaps);
    assert_eq!(r.steps, 4);
    assert_eq!(labels_of(&dir, &rel).overlap_suggestions.len(), 1);
    svc.shutdown();
}

/// Runs the real `autolabel.py --zoom-pairs` of the job on the real GPU with an installed environment;
/// everything else (uv, the weights' download) stays fake.
struct RealScript {
    fake: ZoomFake,
    python: PathBuf,
    weights: PathBuf,
}

impl CommandRunner for RealScript {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        if !cmd.args.iter().any(|a| a == "--zoom-pairs") {
            return self.fake.run(cmd, on_line, cancel);
        }
        let mut real = cmd.clone();
        real.program.clone_from(&self.python);
        let i = real.args.iter().position(|a| a == "--rfdetr").unwrap();
        real.args[i + 1] = self.weights.clone().into_os_string();
        crate::resources::export::SystemRunner.run(&real, on_line, cancel)
    }
}

/// The whole job but the environment stage on a copy of a dataset, with the real script and model:
/// `BOP_OVERLAPS_COPY=<dataset copy> BOP_TRAIN_PYTHON=<train-env python>
/// BOP_RFDETR_WEIGHTS=<rf-detr-large.pth> cargo test --lib real_script -- --ignored --nocapture`.
/// Never point it at a dataset someone is reviewing.
#[test]
#[ignore = "needs a GPU, an installed training environment and a copy of a dataset"]
fn real_script_on_a_copy() {
    let var = |k: &str| PathBuf::from(std::env::var_os(k).unwrap());
    let t = tmp("real");
    let mut job = plan(&t.0, "");
    job.dataset_dir = var("BOP_OVERLAPS_COPY");
    job.free_commit = crate::system_info::commit;
    let fake = RealScript {
        fake: ZoomFake::default(),
        python: var("BOP_TRAIN_PYTHON"),
        weights: var("BOP_RFDETR_WEIGHTS"),
    };
    let started = std::time::Instant::now();
    let mut last = 0;
    let out = job
        .run(
            TaskContext {
                runner: &fake,
                fetcher: &fake.fake,
            },
            &mut |r| match r {
                TaskReport::Stage {
                    stage,
                    percent,
                    detail,
                    ..
                } if percent / 10 != last => {
                    last = percent / 10;
                    println!("{percent}% {stage}: {detail}");
                }
                TaskReport::Line(l) if !l.trim().is_empty() => println!("{l}"),
                _ => {}
            },
            &AtomicBool::new(false),
        )
        .unwrap();
    println!("{:.1} s: {}", started.elapsed().as_secs_f64(), out.summary);
    println!("{}", out.result);
}
