//! The auto-label stage machine ([`AutolabelJob`], run on the training queue as a [`Task`]):
//! select the frames (the Review queue's order and filters) -> set up the training environment
//! -> download the labelers (MM-Grounding-DINO, RF-DETR L, Qwen3-VL, SigLIP 2; SHA-256 pinned) ->
//! label (`scripts/autolabel.py`, `@autolabel` events, proposals as JSONL) -> triage and merge
//! into the label files (here, [`super::triage`]; every labelled frame gets an `auto_run`
//! marker, so a later run of the same labelers skips it unless `again=1`) and queue the changed
//! frames for review. With
//! `with_model` the labelling is the service's own pipeline ([`super::relabel`]) and stages 2 and
//! 3 are skipped.

use super::relabel::{Detector, Image};
use super::triage::{self, FrameProposals};
use super::{AutolabelRequest, Category};
use crate::collect::labels::{self, Frame, LabelFile, Triage, queue};
use crate::config::Config;
use crate::resources::catalog::{Platform, Resource};
use crate::resources::export::train::TrainPaths;
use crate::training::capability::Accelerator;
use crate::training::pyenv::{self, check, stage};
use crate::training::task::{Task, TaskContext, TaskKind, TaskOutcome, TaskReport};
use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

/// Stages of a run.
pub const STAGES: [&str; 5] = [
    "selecting frames",
    "setting up the training environment",
    "downloading the labelers",
    "labelling",
    "writing labels",
];

/// A planned run ([`super::plan_autolabel`]).
#[derive(Clone)]
pub struct AutolabelJob {
    pub id: String,
    pub request: AutolabelRequest,
    pub dataset_dir: PathBuf,
    pub classes: Vec<String>,
    /// Prompt phrases per class.
    pub synonyms: BTreeMap<String, Vec<String>>,
    pub paths: TrainPaths,
    pub platform: Platform,
    pub accelerator: Option<Accelerator>,
    pub timeout: Duration,
    /// `<training dir>/autolabel/<id>`: the frame list and the proposals (deleted at the end).
    pub work: PathBuf,
    pub config: Box<Config>,
    /// Detector of `with_model` runs (None: [`super::relabel::LocalDetector`] on the config).
    pub detector: Option<Arc<dyn Detector>>,
}

impl std::fmt::Debug for AutolabelJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutolabelJob")
            .field("id", &self.id)
            .field("request", &self.request)
            .field("dataset_dir", &self.dataset_dir)
            .finish_non_exhaustive()
    }
}

/// A selected frame.
#[derive(Debug, Clone)]
struct Selected {
    rel: String,
    jpg: PathBuf,
    scene: String,
    width: u32,
    height: u32,
}

/// What [`AutolabelJob::write_frame`] does with a frame's updated label file.
#[derive(Debug, Clone, Copy, Default)]
struct FrameWrite {
    /// Write it.
    write: bool,
    /// The boxes changed: queue the frame for review.
    queue: bool,
}

impl FrameWrite {
    /// Write when `changed` and queue it.
    fn changed(changed: bool) -> Self {
        Self {
            write: changed,
            queue: changed,
        }
    }
}

/// The report of a finished run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AutolabelReport {
    pub frames: usize,
    /// Frames with at least one proposal.
    pub with_boxes: usize,
    pub proposals: usize,
    pub accept: usize,
    pub review: usize,
    /// Pending boxes added to label files.
    pub added: usize,
    /// Proposals an existing box already covered.
    pub dropped: usize,
    /// Existing boxes that got the auto-labeler's evidence.
    pub annotated: usize,
    /// Pending boxes of another class that got the labelers' class as a suggestion.
    pub class_suggestions: usize,
    /// Matching frames left out: these labelers looked at them already (`again=1` to redo).
    pub skipped_done: usize,
    /// Reviewed or skipped frames that went back to review.
    pub reopened: usize,
    /// Self-training: frames whose reviewed labels the model disagrees with.
    pub disagreements: usize,
    pub by_label: BTreeMap<String, usize>,
    /// The VLM said "no" on these boxes.
    pub vlm_objections: usize,
    pub labeler: String,
    pub device: Option<String>,
    pub seconds: f64,
    /// The first changed frames (`<scene>/<camera>/<date>/<stem>.jpg`).
    pub sample: Vec<String>,
    pub warnings: Vec<String>,
}

impl AutolabelJob {
    fn step(&self, report: &mut dyn FnMut(TaskReport<'_>), n: usize, pct: u8, detail: &str) {
        let name = STAGES.get(n.saturating_sub(1)).copied().unwrap_or("");
        stage(report, name, n, STAGES.len(), pct, detail);
    }

    /// The labelers of a labelers-mode run (`mm-gdino-tiny + rfdetr-large + ...`), as recorded
    /// in the report and the frames' `auto_run`.
    fn labeler(&self) -> String {
        let r = &self.request;
        [
            r.gdino.map(|g| g.key().to_string()),
            r.rfdetr.then(|| "rfdetr-large".to_string()),
            r.vlm.map(|v| v.key().to_string()),
            r.siglip.then(|| "siglip2-base".to_string()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" + ")
    }

    /// The frames to label, in queue order, and how many matching frames were left out because
    /// `skip_labeler` (Some in labelers mode without `again=1`) already looked at them.
    fn select(&self, skip_labeler: Option<&str>) -> Result<(Vec<Selected>, usize)> {
        let f = &self.request.frames;
        let built = queue::build(&self.dataset_dir, &f.filters()?);
        let mut out = Vec::new();
        let mut skipped = 0;
        for item in built.items {
            if out.len() >= f.limit {
                break;
            }
            let Some(frame) = Frame::resolve(&self.dataset_dir, &item.frame) else {
                continue;
            };
            let Ok(rec) = frame.record() else { continue };
            if !f.accepts_source(rec.source.as_deref()) {
                continue;
            }
            if let Some(l) = skip_labeler
                && frame
                    .labels()
                    .ok()
                    .flatten()
                    .and_then(|x| x.auto_run)
                    .is_some_and(|r| r.labeler == l)
            {
                skipped += 1;
                continue;
            }
            out.push(Selected {
                rel: item.frame,
                jpg: frame.jpg,
                scene: rec.scene,
                width: rec.width,
                height: rec.height,
            });
        }
        Ok((out, skipped))
    }

    /// Update one frame's label file (the pre-fill when it has none) under the label write lock
    /// ([`labels::update`]): `update` returns whether to write it and whether the frame changed
    /// for the reviewer (then it is queued for review).
    fn write_frame(
        &self,
        sel: &Selected,
        rep: &mut AutolabelReport,
        changed: &mut Vec<String>,
        update: impl FnOnce(&mut LabelFile, &mut AutolabelReport) -> FrameWrite,
    ) -> Result<()> {
        let frame = Frame::resolve(&self.dataset_dir, &sel.rel)
            .ok_or_else(|| anyhow!("{} is gone", sel.rel))?;
        let rec = frame.record()?;
        let mut outcome = FrameWrite::default();
        let mut reopened = false;
        let written = labels::update(&frame.labels, |cur| {
            let mut file = cur.unwrap_or_else(|| labels::prefill(&rec));
            let before = file.status;
            outcome = update(&mut file, rep);
            reopened = before != file.status && file.status == labels::Status::Unreviewed;
            Ok(outcome.write.then_some(file))
        })?;
        if written {
            rep.reopened += usize::from(reopened);
            if outcome.queue {
                changed.push(sel.rel.clone());
            }
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one linear stage machine reads best in one place"
    )]
    fn stages(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<TaskOutcome> {
        let started = Instant::now();
        let r = &self.request;
        let mut rep = AutolabelReport::default();

        // 1. Frames.
        self.step(report, 1, 0, &r.frames.describe());
        let labeler = self.labeler();
        let skip = (r.with_model.is_none() && !r.frames.again()).then_some(labeler.as_str());
        let (frames, skipped) = self.select(skip)?;
        rep.skipped_done = skipped;
        if frames.is_empty() && skipped > 0 {
            bail!(
                "all {skipped} frame(s) matching {} were auto-labelled by {labeler} already; add \
                 again=1 to the frame filter (Review page: tick \"Label again\") to label them again",
                r.frames.describe()
            );
        }
        if frames.is_empty() {
            bail!(
                "no frame matches {} in {}; import footage or send frames to review first",
                r.frames.describe(),
                self.dataset_dir.display()
            );
        }
        rep.frames = frames.len();
        self.step(report, 1, 100, &format!("{} frames", frames.len()));
        check(cancel)?;
        let _lock =
            crate::resources::export::train::TrainLock::try_acquire(&self.paths.train_lock_file())?;
        let mut changed: Vec<String> = Vec::new();

        if let Some(model) = &r.with_model {
            rep.labeler = format!("model {model}");
            self.step(report, 2, 100, "not needed");
            self.step(report, 3, 100, "not needed");
            self.step(
                report,
                4,
                0,
                &format!("running {model} on {} frames", frames.len()),
            );
            let images: Vec<Image> = frames
                .iter()
                .map(|f| Image {
                    frame: f.rel.clone(),
                    path: f.jpg.clone(),
                    width: f.width,
                    height: f.height,
                })
                .collect();
            let local;
            let det: &dyn Detector = match &self.detector {
                Some(d) => d.as_ref(),
                None => {
                    local = super::relabel::LocalDetector {
                        config: (*self.config).clone(),
                        runtimes: None,
                    };
                    &local
                }
            };
            let dets = det.detect(model, &images, r.threshold, cancel)?;
            self.step(report, 4, 100, &format!("{model}: done"));
            check(cancel)?;
            for (i, f) in frames.iter().enumerate() {
                if i % 10 == 0 {
                    let pct = u8::try_from(i * 100 / frames.len().max(1)).unwrap_or(100);
                    self.step(report, 5, pct, &format!("{i} of {}", frames.len()));
                }
                let d = dets.get(&f.rel).cloned().unwrap_or_default();
                if !d.is_empty() {
                    rep.with_boxes += 1;
                }
                rep.proposals += d.len();
                rep.review += d.len();
                for x in &d {
                    *rep.by_label.entry(x.label.clone()).or_default() += 1;
                }
                let res = self.write_frame(f, &mut rep, &mut changed, |file, rep| {
                    let (m, why) =
                        triage::apply_relabel(file, &d, model, &self.classes, f.width, f.height);
                    rep.added += m.added;
                    rep.dropped += m.dropped;
                    if why.is_some() {
                        rep.disagreements += 1;
                    }
                    FrameWrite::changed(m.added > 0 || why.is_some())
                });
                if let Err(e) = res {
                    rep.warnings.push(format!("{}: {e:#}", f.rel));
                }
            }
        } else {
            rep.labeler = labeler.clone();
            // 2. Environment.
            let accel = self
                .accelerator
                .as_ref()
                .ok_or_else(|| anyhow!("the auto-labelers need an NVIDIA GPU or Apple silicon"))?;
            rep.device = Some(accel.device_arg());
            pyenv::setup(
                &self.paths,
                self.platform,
                accel,
                ctx.runner,
                ctx.fetcher,
                report,
                (STAGES[1], 2, STAGES.len()),
                cancel,
            )?;
            let (_, script) = crate::resources::export::train::write_task_scripts(&self.paths)?;
            // The script's own tests of its fusion code, and the imports.
            self.step(report, 2, 95, "checking the auto-labeler");
            let mut passed = None;
            pyenv::run_script(
                ctx.runner,
                &self.paths,
                self.platform,
                &self.paths.scripts,
                Duration::from_secs(15 * 60),
                vec![script.clone().into_os_string(), "--self-check".into()],
                "autolabel",
                &mut |ev, _| {
                    if ev.get("event").and_then(serde_json::Value::as_str) == Some("self_test") {
                        passed = ev.get("passed").and_then(serde_json::Value::as_u64);
                    }
                },
                report,
                cancel,
            )
            .context("the auto-labeler's self-check failed")?;
            if passed.is_none() {
                bail!("the auto-labeler's self-test did not run; the script is damaged");
            }
            check(cancel)?;
            // 3. Weights.
            let res: Vec<&'static Resource> = r.resources();
            let list: Vec<(&'static Resource, PathBuf)> = res
                .iter()
                .map(|x| (*x, pyenv::weights_dir(&self.paths, x)))
                .collect();
            pyenv::check_large(&list, r.allow_large)?;
            pyenv::fetch_missing(
                ctx.fetcher,
                &list,
                report,
                (STAGES[2], 3, STAGES.len()),
                cancel,
            )?;
            check(cancel)?;
            // 4. Label.
            std::fs::create_dir_all(&self.work)
                .with_context(|| format!("creating {}", self.work.display()))?;
            let list_path = self.work.join("frames.jsonl");
            let mut text = String::new();
            for f in &frames {
                text.push_str(&serde_json::to_string(&serde_json::json!({
                    "frame": f.rel, "path": f.jpg, "width": f.width, "height": f.height,
                    "scene": f.scene,
                }))?);
                text.push('\n');
            }
            std::fs::write(&list_path, text)?;
            let syn = self.work.join("synonyms.json");
            std::fs::write(&syn, serde_json::to_vec(&self.synonyms)?)?;
            let cats: BTreeMap<&str, Category> = self
                .classes
                .iter()
                .map(|c| (c.as_str(), Category::of(c)))
                .collect();
            let cat_path = self.work.join("categories.json");
            std::fs::write(&cat_path, serde_json::to_vec(&cats)?)?;
            let out = self.work.join("proposals.jsonl");
            let dir_of = |key: &str| -> Result<OsString> {
                let r = list
                    .iter()
                    .find(|(x, _)| x.id.ends_with(key))
                    .ok_or_else(|| anyhow!("internal: {key} not planned"))?;
                Ok(r.1.clone().into_os_string())
            };
            let mut args: Vec<OsString> = vec![
                script.into_os_string(),
                "--frames".into(),
                list_path.into_os_string(),
                "--out".into(),
                out.clone().into_os_string(),
                "--classes".into(),
                serde_json::to_string(&self.classes)?.into(),
                "--synonyms".into(),
                syn.into_os_string(),
                "--categories".into(),
                cat_path.into_os_string(),
                "--device".into(),
                accel.device_arg().into(),
                "--gdino-threshold".into(),
                r.gdino_threshold.to_string().into(),
                "--rfdetr-threshold".into(),
                r.rfdetr_threshold.to_string().into(),
            ];
            if let Some(g) = r.gdino {
                args.push("--gdino".into());
                args.push(dir_of(g.key())?);
            }
            if r.rfdetr {
                let rf = list
                    .iter()
                    .find(|(x, _)| x.id == "train-base:rfdetr-large")
                    .ok_or_else(|| anyhow!("internal: RF-DETR L not planned"))?;
                let file =
                    rf.0.parts
                        .first()
                        .ok_or_else(|| anyhow!("RF-DETR L has no file"))?
                        .file_name;
                args.push("--rfdetr".into());
                args.push(rf.1.join(file).into_os_string());
            }
            if let Some(v) = r.vlm {
                args.push("--vlm".into());
                args.push(dir_of(v.key())?);
                args.push("--vlm-name".into());
                args.push(v.key().into());
            }
            if r.siglip {
                args.push("--siglip".into());
                args.push(dir_of("siglip2-base")?);
            }
            let total = frames.len();
            let mut done_ev = None;
            pyenv::run_script(
                ctx.runner,
                &self.paths,
                self.platform,
                &self.work,
                self.timeout,
                args,
                "autolabel",
                &mut |ev, report| match ev.get("event").and_then(serde_json::Value::as_str) {
                    Some("frame") => {
                        let i = ev
                            .get("index")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        let pct = u8::try_from(i * 100 / u64::try_from(total.max(1)).unwrap_or(1))
                            .unwrap_or(100);
                        stage(
                            report,
                            STAGES[3],
                            4,
                            STAGES.len(),
                            pct.min(99),
                            &format!("{i} of {total} frames"),
                        );
                    }
                    Some("model") => stage(
                        report,
                        STAGES[3],
                        4,
                        STAGES.len(),
                        0,
                        &format!(
                            "loaded {}",
                            ev.get("name")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("?")
                        ),
                    ),
                    Some("done") => done_ev = Some(ev.clone()),
                    _ => {}
                },
                report,
                cancel,
            )
            .context("the auto-labeler failed")?;
            if done_ev.is_none() {
                bail!("the auto-labeler finished without its summary; the script is damaged");
            }
            check(cancel)?;
            // 5. Triage and write.
            let text = std::fs::read_to_string(&out)
                .with_context(|| format!("reading {}", out.display()))?;
            let by_rel: BTreeMap<&str, &Selected> =
                frames.iter().map(|f| (f.rel.as_str(), f)).collect();
            let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            for (i, line) in lines.iter().enumerate() {
                if i % 10 == 0 {
                    let pct = u8::try_from(i * 100 / lines.len().max(1)).unwrap_or(100);
                    self.step(report, 5, pct, &format!("{i} of {}", lines.len()));
                }
                let fp: FrameProposals = match serde_json::from_str(line) {
                    Ok(x) => x,
                    Err(e) => {
                        rep.warnings
                            .push(format!("a proposals line does not parse: {e}"));
                        continue;
                    }
                };
                let Some(sel) = by_rel.get(fp.frame.as_str()) else {
                    rep.warnings
                        .push(format!("{}: not a selected frame; ignored", fp.frame));
                    continue;
                };
                let boxes = triage::to_boxes(
                    &fp.boxes,
                    &self.classes,
                    &sel.scene,
                    sel.width,
                    sel.height,
                    r.accept_score,
                );
                if !boxes.is_empty() {
                    rep.with_boxes += 1;
                }
                rep.proposals += boxes.len();
                for b in &boxes {
                    *rep.by_label.entry(b.label.clone()).or_default() += 1;
                    match b.auto.triage {
                        Some(Triage::Accept) => rep.accept += 1,
                        _ => rep.review += 1,
                    }
                    if b.auto.vlm.as_ref().is_some_and(|v| !v.ok) {
                        rep.vlm_objections += 1;
                    }
                }
                // Every frame the labelers returned gets the marker, boxes or not.
                let marker = labels::AutoRun {
                    at: crate::collect::utc_rfc3339(std::time::SystemTime::now()),
                    labeler: labeler.clone(),
                };
                let res = self.write_frame(sel, &mut rep, &mut changed, |file, rep| {
                    file.auto_run = Some(marker);
                    if boxes.is_empty() {
                        return FrameWrite {
                            write: true,
                            queue: false,
                        };
                    }
                    let a = triage::apply(file, boxes);
                    rep.added += a.merge.added;
                    rep.dropped += a.merge.dropped;
                    rep.annotated += a.annotated;
                    rep.class_suggestions += a.class_suggestions;
                    FrameWrite {
                        write: true,
                        queue: a.merge.added > 0 || a.annotated > 0 || a.class_suggestions > 0,
                    }
                });
                if let Err(e) = res {
                    rep.warnings.push(format!("{}: {e:#}", sel.rel));
                }
            }
        }
        crate::collect::layout::append_review(&self.dataset_dir, &changed)
            .context("queueing the labelled frames for review")?;
        rep.sample = changed.iter().take(20).cloned().collect();
        rep.seconds = (started.elapsed().as_secs_f64() * 10.0).round() / 10.0;
        let _ = crate::resources::manager::remove_any(&self.work);
        self.step(report, 5, 100, &format!("{} frames changed", changed.len()));
        let summary = if r.with_model.is_some() {
            format!(
                "{} frames re-labelled by {}: {} disagree with their reviewed labels, {} boxes \
                 added for review",
                rep.frames, rep.labeler, rep.disagreements, rep.added
            )
        } else {
            let mut s = format!(
                "{} frames labelled ({}): {} boxes ({} accept, {} review), {} added to label \
                 files, {} matched existing boxes, {} class suggestions on existing boxes; {} \
                 frames queued for review",
                rep.frames,
                rep.labeler,
                rep.proposals,
                rep.accept,
                rep.review,
                rep.added,
                rep.dropped,
                rep.class_suggestions,
                changed.len()
            );
            if rep.skipped_done > 0 {
                s.push_str(&format!(
                    "; {} frames skipped (labelled by these labelers before; again=1 to redo)",
                    rep.skipped_done
                ));
            }
            s
        };
        Ok(TaskOutcome {
            summary,
            result: serde_json::to_value(&rep)?,
        })
    }
}

impl Task for AutolabelJob {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> TaskKind {
        TaskKind::Autolabel
    }
    fn title(&self) -> String {
        match &self.request.with_model {
            Some(m) => format!("re-label with {m} ({})", self.request.frames.describe()),
            None => format!("auto-label ({})", self.request.frames.describe()),
        }
    }
    fn steps(&self) -> usize {
        STAGES.len()
    }
    fn run(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<TaskOutcome> {
        let r = self.stages(ctx, report, cancel);
        if r.is_err() {
            let _ = crate::resources::manager::remove_any(&self.work);
        }
        r
    }
}

#[cfg(test)]
mod tests;
