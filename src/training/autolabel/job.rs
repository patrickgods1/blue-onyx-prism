//! The auto-label stage machine ([`AutolabelJob`], run on the training queue as a [`Task`]):
//! select the frames (the Review queue's order and filters) -> set up the training environment
//! -> download the labelers (MM-Grounding-DINO, RF-DETR L, Qwen3-VL, SigLIP 2; SHA-256 pinned) ->
//! label (`scripts/autolabel.py`, `@autolabel` events, proposals as JSONL) -> triage and merge
//! into the label files (here, [`super::triage`]; every labelled frame gets an `auto_run`
//! marker, so a later run of the same labelers skips it unless `again=1`) and queue the changed
//! frames for review. With
//! `with_model` the stages are [`MODEL_STAGES`]: select the frames -> install the model (a
//! catalog model not on disk yet: downloaded, or exported with the export toolchain) -> add it
//! to the config disabled (a local or catalog model, through the [`ConfigWriter`]) -> label with
//! the service's own pipeline ([`super::relabel`]) -> map, triage and merge
//! ([`triage::model_boxes`], [`triage::apply_relabel`]); every frame it looked at gets the
//! `model:<name>@<device>` marker.
//!
//! A cancel while labelling with [`AutolabelRequest::keep_on_cancel`] (the default) is not a
//! failure: the frames finished so far (the script's flushed `proposals.jsonl`, the model's
//! finished frames) go through the same triage and write stage, markers included, and the run
//! returns `Ok` with a "stopped by Cancel" summary; a rerun skips them by their marker.

use super::models::{self, ConfigWriter, Install, ModelPlan, Target};
use super::relabel::{Detector, Image};
use super::triage::{self, FrameProposals};
use super::{AutolabelRequest, Category};
use crate::collect::labels::{self, Frame, LabelFile, Triage, queue};
use crate::config::{Config, ModelConfig};
use crate::resources::catalog::{Platform, Resource, ResourceKind};
use crate::resources::export::train::TrainPaths;
use crate::training::capability::Accelerator;
use crate::training::pyenv::{self, check, stage};
use crate::training::task::{Task, TaskContext, TaskKind, TaskOutcome, TaskRecord, TaskReport};
use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
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

fn mb(b: u64) -> String {
    format!("{:.1}", b as f64 / 1e6)
}

/// Stages of a run with a model.
pub const MODEL_STAGES: [&str; 5] = [
    "selecting frames",
    "installing the model",
    "adding the model to the config",
    "labelling",
    "writing labels",
];

/// Whether `e` is a cancel ([`crate::resources::export::Cancelled`] or the benchmark's).
fn is_cancel(e: &anyhow::Error) -> bool {
    e.chain()
        .any(|c| c.is::<crate::resources::export::Cancelled>())
        || crate::benchmark::is_cancelled(e)
}

fn cancelled(cancel: &AtomicBool) -> bool {
    cancel.load(std::sync::atomic::Ordering::Relaxed)
}

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
    /// The resolved `with_model` ([`models::resolve`]).
    pub model: Option<ModelPlan>,
    /// Detector of `with_model` runs (None: [`super::relabel::LocalDetector`] on the config).
    pub detector: Option<Arc<dyn Detector>>,
    /// Saves the config entry of a local or catalog model (None: the run uses the entry without
    /// saving it, with a warning).
    pub config_writer: Option<Arc<dyn ConfigWriter>>,
    /// The import this run follows ([`super::follow_import`]): it labels that import's frames
    /// once the import is done, and is not run when it failed, was cancelled or wrote no frame.
    pub after: Option<String>,
    /// Steps after a finished run ([`crate::training::finish`]).
    pub finish: Option<crate::training::finish::Finish>,
}

impl std::fmt::Debug for AutolabelJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutolabelJob")
            .field("id", &self.id)
            .field("request", &self.request)
            .field("dataset_dir", &self.dataset_dir)
            .field("after", &self.after)
            .field("finish", &self.finish)
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
    /// Boxes per class (the class list's spelling).
    pub by_label: BTreeMap<String, usize>,
    /// Proposals of a label that is no class of the run, also after mapping the spelling.
    pub dropped_other_class: usize,
    /// Labels given the class list's spelling: model label -> class (`motorbike` ->
    /// `motorcycle`).
    pub renamed: BTreeMap<String, String>,
    /// The VLM said "no" on these boxes.
    pub vlm_objections: usize,
    pub labeler: String,
    /// The device that ran (a model: its canonical spec; the labelers: the training device).
    pub device: Option<String>,
    /// A run with a model: the device it asked for (the run's `device`, else the model's).
    pub requested_device: Option<String>,
    /// A run with a model: its id and what the install and config stages did.
    pub model: Option<String>,
    pub installed: Option<String>,
    pub config_entry: Option<String>,
    pub seconds: f64,
    /// The first changed frames (`<scene>/<camera>/<date>/<stem>.jpg`).
    pub sample: Vec<String>,
    pub warnings: Vec<String>,
    /// Stopped by Cancel while labelling (`keep_on_cancel`): only `frames_done` of
    /// `frames_total` frames were labelled and written.
    pub cancelled: bool,
    pub frames_done: usize,
    pub frames_total: usize,
}

impl AutolabelJob {
    fn stage_names(&self) -> &'static [&'static str; 5] {
        if self.model.is_some() {
            &MODEL_STAGES
        } else {
            &STAGES
        }
    }

    fn step(&self, report: &mut dyn FnMut(TaskReport<'_>), n: usize, pct: u8, detail: &str) {
        let names = self.stage_names();
        let name = names.get(n.saturating_sub(1)).copied().unwrap_or("");
        stage(report, name, n, names.len(), pct, detail);
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
    /// `skip_labeler` (Some in labelers mode without `again=1`) already looked at them for every
    /// class of this run ([`labels::AutoRun::covers`]; a run for a class it did not look for
    /// labels the frame again).
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
                    .is_some_and(|r| r.labeler == l && r.covers(&self.classes))
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
            // The pre-fill keeps the whole collect class list, not only this run's classes.
            let mut file =
                cur.unwrap_or_else(|| labels::prefill(&rec, &self.config.collect.classes));
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
        let labeler = match &self.model {
            Some(m) => m.labeler(),
            None => self.labeler(),
        };
        let skip = (!r.frames.again()).then_some(labeler.as_str());
        let (frames, skipped) = self.select(skip)?;
        rep.skipped_done = skipped;
        if frames.is_empty() && skipped > 0 {
            bail!(
                "all {skipped} frame(s) matching {} were auto-labelled by {labeler} for these \
                 classes already; add again=1 to the frame filter (Review page: tick \"Label \
                 again\") to label them again",
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
        // `stopped`: a cancel while labelling with `keep_on_cancel`; the run wrote the
        // `frames_done` frames finished so far.
        let (stopped, frames_done) = if let Some(plan) = &self.model {
            self.run_model(
                plan,
                &frames,
                &labeler,
                ctx,
                report,
                cancel,
                &mut rep,
                &mut changed,
            )?
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
                let batch = if r.vlm_batch { super::VLM_BATCH } else { 1 };
                args.push("--vlm-batch".into());
                args.push(batch.to_string().into());
                if let Some(m) = r.vlm_min_score {
                    args.push("--vlm-min-score".into());
                    args.push(m.to_string().into());
                }
            }
            if r.prefetch {
                args.push("--prefetch".into());
            }
            if r.siglip {
                args.push("--siglip".into());
                args.push(dir_of("siglip2-base")?);
            }
            let total = frames.len();
            let mut done_ev = None;
            let labelled = pyenv::run_script(
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
            );
            // A cancel kills the script; with `keep_on_cancel` the frames it finished (it
            // flushes `proposals.jsonl` after each) are written below like a finished run's.
            let stopped = match labelled {
                Err(e) if r.keep_on_cancel && is_cancel(&e) => true,
                Err(e) => return Err(e.context("the auto-labeler failed")),
                Ok(()) => r.keep_on_cancel && cancelled(cancel),
            };
            if !stopped {
                if done_ev.is_none() {
                    bail!("the auto-labeler finished without its summary; the script is damaged");
                }
                check(cancel)?;
            }
            // 5. Triage and write (no cancel checks: a stopped run writes what it has).
            let text = match std::fs::read_to_string(&out) {
                Err(e) if stopped && e.kind() == std::io::ErrorKind::NotFound => String::new(),
                t => t.with_context(|| format!("reading {}", out.display()))?,
            };
            let by_rel: BTreeMap<&str, &Selected> =
                frames.iter().map(|f| (f.rel.as_str(), f)).collect();
            let mut done: BTreeSet<&str> = BTreeSet::new();
            let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
            for (i, line) in lines.iter().enumerate() {
                if i % 10 == 0 {
                    let pct = u8::try_from(i * 100 / lines.len().max(1)).unwrap_or(100);
                    self.step(report, 5, pct, &format!("{i} of {}", lines.len()));
                }
                let fp: FrameProposals = match serde_json::from_str(line) {
                    Ok(x) => x,
                    // The line the killed script was writing.
                    Err(_) if stopped && i + 1 == lines.len() => continue,
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
                done.insert(sel.rel.as_str());
                let out = triage::to_boxes_counted(
                    &fp.boxes,
                    &self.classes,
                    &sel.scene,
                    sel.width,
                    sel.height,
                    r.accept_score,
                );
                rep.dropped_other_class += out.other_class;
                rep.renamed.extend(out.renamed);
                let boxes = out.boxes;
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
                    classes: self.classes.clone(),
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
            (stopped, done.len())
        };
        if stopped && frames_done == 0 {
            // Nothing finished: a plain cancel.
            return Err(anyhow!(crate::resources::export::Cancelled));
        }
        rep.cancelled = stopped;
        rep.frames_total = frames.len();
        rep.frames_done = frames_done;
        rep.frames = frames_done;
        crate::collect::layout::append_review(&self.dataset_dir, &changed)
            .context("queueing the labelled frames for review")?;
        rep.sample = changed.iter().take(20).cloned().collect();
        rep.seconds = (started.elapsed().as_secs_f64() * 10.0).round() / 10.0;
        let _ = crate::resources::manager::remove_any(&self.work);
        self.step(report, 5, 100, &format!("{} frames changed", changed.len()));
        let summary = if stopped {
            let n = crate::training::import::preview::thousands;
            format!(
                "stopped by Cancel after {} of {} frames; those {} were saved for review ({} \
                 boxes: {} accept, {} review; {} added to label files; {} frames queued for \
                 review); run the same auto-label again to label the rest (it skips the frames \
                 done)",
                n(frames_done),
                n(frames.len()),
                n(frames_done),
                n(rep.proposals),
                n(rep.accept),
                n(rep.review),
                n(rep.added),
                n(changed.len())
            )
        } else if let Some(plan) = &self.model {
            let mut s = format!(
                "{} frames labelled by {} on {}: {} boxes ({} accept, {} review), {} added to \
                 label files, {} matched existing boxes; {} frames disagree with their reviewed \
                 labels; {} frames queued for review",
                rep.frames,
                plan.name,
                rep.device.as_deref().unwrap_or(&plan.device),
                rep.proposals,
                rep.accept,
                rep.review,
                rep.added,
                rep.dropped,
                rep.disagreements,
                changed.len()
            );
            if rep.dropped_other_class > 0 {
                s.push_str(&format!(
                    "; {} detections of classes outside this run left out",
                    rep.dropped_other_class
                ));
            }
            if rep.skipped_done > 0 {
                s.push_str(&format!(
                    "; {} frames skipped (labelled by this model on this device before; again=1 \
                     to redo)",
                    rep.skipped_done
                ));
            }
            s
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

impl AutolabelJob {
    /// Stages 2 to 5 of a run with a model; returns whether a cancel stopped the labelling
    /// (`keep_on_cancel`) and the frames labelled.
    #[allow(
        clippy::too_many_arguments,
        reason = "the stage machine's state, passed down once"
    )]
    fn run_model(
        &self,
        plan: &ModelPlan,
        frames: &[Selected],
        labeler: &str,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
        rep: &mut AutolabelReport,
        changed: &mut Vec<String>,
    ) -> Result<(bool, usize)> {
        let r = &self.request;
        rep.labeler = labeler.to_string();
        rep.model = Some(plan.id.clone());
        rep.requested_device = Some(plan.device.clone());
        // 2. Install.
        match (plan.install, plan.resource()) {
            (Some(how), Some(res)) => {
                let msg = self.install(res, how, ctx, report, cancel)?;
                self.step(report, 2, 100, &msg);
                rep.installed = Some(msg);
            }
            _ => self.step(report, 2, 100, "installed"),
        }
        check(cancel)?;
        // 3. Config entry.
        let (config, entry) = self.config_entry(plan, rep)?;
        self.step(
            report,
            3,
            100,
            rep.config_entry.as_deref().unwrap_or("in the config"),
        );
        check(cancel)?;
        // 4. Label.
        let device_text = r.device.as_deref().unwrap_or(&plan.device);
        self.step(
            report,
            4,
            0,
            &format!(
                "running {} on {} frames ({device_text})",
                plan.name,
                frames.len()
            ),
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
                    config: config.clone(),
                    runtimes: None,
                };
                &local
            }
        };
        let dets = det
            .detect(
                &config,
                &entry,
                r.device.as_deref(),
                &images,
                r.threshold,
                cancel,
            )
            .map_err(|e| {
                if entry.family == crate::model::ModelFamilyKind::Auto
                    && e.downcast_ref::<crate::resources::export::Cancelled>()
                        .is_none()
                {
                    e.context(format!(
                        "{} has family auto (detected from its outputs); if that is what \
                         failed, set its family on the Config page (Models card) and run again",
                        plan.name
                    ))
                } else {
                    e
                }
            })?;
        if !dets.device.is_empty() {
            rep.device = Some(dets.device.clone());
        }
        // A cancel with `keep_on_cancel`: the frames the model finished are written below
        // (no cancel checks there), the others are left as they are.
        let stopped = r.keep_on_cancel && (dets.cancelled || cancelled(cancel));
        if dets.cancelled && !stopped {
            return Err(anyhow!(crate::resources::export::Cancelled));
        }
        let todo: Vec<&Selected> = frames
            .iter()
            .filter(|f| !dets.cancelled || dets.frames.contains_key(&f.rel))
            .collect();
        self.step(report, 4, 100, &format!("{}: done", plan.name));
        if !stopped {
            check(cancel)?;
        }
        // 5. Map, triage, merge; every frame gets the marker.
        for (i, f) in todo.iter().copied().enumerate() {
            if i % 10 == 0 {
                let pct = u8::try_from(i * 100 / todo.len().max(1)).unwrap_or(100);
                self.step(report, 5, pct, &format!("{i} of {}", todo.len()));
            }
            let d = dets.frames.get(&f.rel).cloned().unwrap_or_default();
            let run = triage::ModelRun {
                model: &plan.name,
                classes: &self.classes,
                scene: &f.scene,
                accept_score: r.model_accept_score,
            };
            let out = triage::model_boxes(&d, &run, f.width, f.height);
            rep.dropped_other_class += out.other_class;
            rep.renamed.extend(out.renamed);
            if !out.boxes.is_empty() {
                rep.with_boxes += 1;
            }
            rep.proposals += out.boxes.len();
            for b in &out.boxes {
                *rep.by_label.entry(b.label.clone()).or_default() += 1;
                match b.auto.triage {
                    Some(Triage::Accept) => rep.accept += 1,
                    _ => rep.review += 1,
                }
            }
            let marker = labels::AutoRun {
                at: crate::collect::utc_rfc3339(std::time::SystemTime::now()),
                labeler: labeler.to_string(),
                classes: self.classes.clone(),
            };
            let res = self.write_frame(f, rep, changed, |file, rep| {
                file.auto_run = Some(marker);
                let (a, why) =
                    triage::apply_relabel(file, &d, out.boxes, &plan.name, &self.classes);
                rep.added += a.merge.added;
                rep.dropped += a.merge.dropped;
                rep.annotated += a.annotated;
                rep.class_suggestions += a.class_suggestions;
                if why.is_some() {
                    rep.disagreements += 1;
                }
                FrameWrite {
                    write: true,
                    queue: a.merge.added > 0
                        || a.annotated > 0
                        || a.class_suggestions > 0
                        || why.is_some(),
                }
            });
            if let Err(e) = res {
                rep.warnings.push(format!("{}: {e:#}", f.rel));
            }
        }
        Ok((stopped, todo.len()))
    }

    /// Download or export catalog model `res` into the models folder (stage 2); the export
    /// toolchain's own stages show as the detail.
    fn install(
        &self,
        res: &'static Resource,
        how: Install,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<String> {
        let config = &self.config;
        if models::catalog_installed(config, res) {
            return Ok(format!("{} is installed", res.title));
        }
        match how {
            Install::Download => {
                let dir = crate::resources::status::target_dir(config, res);
                self.step(report, 2, 0, &format!("downloading {}", res.describe()));
                let mut last = 255u8;
                ctx.fetcher
                    .fetch(
                        res,
                        &dir,
                        &mut |bytes, total| {
                            let pct = u8::try_from(bytes.saturating_mul(100) / total.max(1))
                                .unwrap_or(100)
                                .min(99);
                            if pct != last {
                                last = pct;
                                self.step(
                                    report,
                                    2,
                                    pct,
                                    &format!(
                                        "downloading {}: {} of {} MB",
                                        res.title,
                                        mb(bytes),
                                        mb(total)
                                    ),
                                );
                            }
                        },
                        cancel,
                    )
                    .with_context(|| format!("downloading {}", res.title))?;
                if !models::catalog_installed(config, res) {
                    bail!(
                        "{} was downloaded but its files are not in {}; check the Resources \
                         card",
                        res.title,
                        dir.display()
                    );
                }
                Ok(format!(
                    "downloaded {} into {}",
                    res.describe(),
                    dir.display()
                ))
            }
            Install::Export => {
                if res.kind != ResourceKind::ExportModel {
                    bail!("internal: {} is not exported", res.id);
                }
                let mut job = crate::resources::export::ExportJob::new(
                    res,
                    crate::resources::export::Paths::for_config(config),
                );
                job.platform = self.platform;
                let name = res.model_name().unwrap_or(res.id);
                let mut on = |p: crate::resources::export::Progress<'_>| match p {
                    crate::resources::export::Progress::Stage {
                        stage: st,
                        percent,
                        detail,
                    } => self.step(
                        report,
                        2,
                        st.overall(percent).min(99),
                        &format!(
                            "exporting {name}: {} ({}/{}){}",
                            st.label(),
                            st.number(),
                            crate::resources::export::Stage::ALL.len(),
                            if detail.is_empty() {
                                String::new()
                            } else {
                                format!(": {detail}")
                            }
                        ),
                    ),
                    crate::resources::export::Progress::Line { line, .. } => {
                        report(TaskReport::Line(line));
                    }
                };
                let out = crate::resources::export::run_export(
                    &job,
                    ctx.runner,
                    ctx.fetcher,
                    &mut on,
                    cancel,
                )
                .map_err(|e| {
                    if e.cancelled {
                        anyhow!(crate::resources::export::Cancelled)
                    } else {
                        anyhow!(
                            "export of {name} failed{}: {}",
                            e.stage
                                .map(|s| format!(" while {}", s.label()))
                                .unwrap_or_default(),
                            e.message
                        )
                    }
                })?;
                Ok(format!(
                    "exported {name} ({}) to {}",
                    res.license,
                    out.onnx.display()
                ))
            }
        }
    }

    /// The config and entry the run uses (stage 3): a configured model's, else the model added
    /// disabled through the writer (without one, to this run's copy only).
    fn config_entry(
        &self,
        plan: &ModelPlan,
        rep: &mut AutolabelReport,
    ) -> Result<(Config, ModelConfig)> {
        let config = match &plan.target {
            Target::Config { .. } => (*self.config).clone(),
            Target::Local { .. } | Target::Catalog { .. } => {
                let mut msg = String::new();
                let mut edit = |c: &mut Config| -> Result<()> {
                    msg = models::add_disabled(c, plan)?;
                    Ok(())
                };
                match &self.config_writer {
                    Some(w) => {
                        let (c, saved) = w.update(&mut edit)?;
                        rep.config_entry = Some(format!("{msg}; {saved}"));
                        c
                    }
                    None => {
                        let mut c = (*self.config).clone();
                        edit(&mut c)?;
                        rep.warnings.push(format!(
                            "{} was not saved to the config file (this run only)",
                            plan.name
                        ));
                        rep.config_entry = Some(msg);
                        c
                    }
                }
            }
        };
        let entry = models::entry_of(&config, plan).cloned().ok_or_else(|| {
            anyhow!(
                "{} is not in the config (any more); choose the model again on the Training \
                 page",
                plan.name
            )
        })?;
        Ok((config, entry))
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
        // A run after an import names it instead of its frame filter.
        let frames = match &self.after {
            Some(id) => format!("import {id}"),
            None => self.request.frames.describe(),
        };
        match (&self.model, &self.after) {
            (Some(m), _) => format!("label with {} on {} ({frames})", m.name, m.device),
            (None, Some(_)) => match (&self.request.gpu, &self.accelerator) {
                (Some(_), Some(a)) => {
                    format!("auto-label of {frames} on {}", a.device_arg())
                }
                _ => format!("auto-label of {frames}"),
            },
            // A chosen GPU is named (the automatic pick is the page's "PyTorch on ...").
            (None, None) => match (&self.request.gpu, &self.accelerator) {
                (Some(_), Some(a)) => format!("auto-label on {} ({frames})", a.device_arg()),
                _ => format!("auto-label ({frames})"),
            },
        }
    }
    fn steps(&self) -> usize {
        self.stage_names().len()
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
        let mut out = r?;
        if let Some(fin) = self.finish.as_ref().filter(|f| !f.is_empty())
            && !cancelled(cancel)
            && out.result.get("cancelled") != Some(&serde_json::Value::Bool(true))
        {
            let scope = self.request.frames.import().filter(|i| *i != "any");
            let lines = fin.apply(&self.dataset_dir, scope, &self.config.collect.classes);
            crate::training::finish::append(&mut out, lines);
        }
        Ok(out)
    }
    fn after(&self) -> Option<&str> {
        self.after.as_deref()
    }
    fn skip_reason(&self, dep: &TaskRecord) -> Option<String> {
        crate::training::task::default_skip_reason(dep).or_else(|| {
            let frames = dep.result.as_ref().and_then(|r| r.get("frames"));
            (frames.and_then(serde_json::Value::as_u64) == Some(0))
                .then(|| super::NO_NEW_FRAMES.to_string())
        })
    }
}

#[cfg(test)]
mod tests;
