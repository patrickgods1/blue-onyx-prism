//! The zoom stage of the dataset check: a closer look at its loose labels, after the last fold.
//!
//! A fold model's box alone never overrules a label (fold models train a few epochs at a small
//! input size), so a second, independent and sharper look decides: RF-DETR L on a padded crop of
//! the pair, through the overlap check's own script and runner (`scripts/autolabel.py
//! --zoom-pairs`, [`crate::training::overlaps::zoom`]), unchanged: one crop per candidate, the
//! union of the label `L` and the fold model's detection `D` ([`Loose::union`]), padded by the
//! script's `--zoom-pad`. [`super::judge::tighten`] decides from both looks.
//!
//! - **Candidates** ([`FrameCandidates`]): what each fold's writes judged loose and worth a look
//!   ([`super::judge::Judged::candidates`]), with the fold model's detections of the frame, appended to
//!   `<work>/`[`CANDIDATES_FILE`] as the folds finish. The work folder goes at the end of the
//!   check however it ends, so a crash leaves them for nobody.
//! - **The look** runs once, after the last fold, on every candidate of the check, holding the
//!   training lock the check holds. It is one step of the job ("looking closer at N loose
//!   labels"), cancellable and pausable like the overlap check (the queue's runner suspends the
//!   script). A Cancel before it (during the folds) skips it: Cancel means stop.
//! - **The write** ([`super::write::write_tighten`]) judges each frame afresh under the label
//!   write lock: the frame still reviewed, its candidates found again from the fold model's
//!   detections on the file as it is now, each matched to its crop's answer ([`Loose::same_pair`]).
//!   `tighten` is among the kinds written only for frames whose candidates the look answered, so a
//!   check without the stage, a failed or cancelled look, or a crop that never finished, never
//!   deletes an earlier `tighten`; an open one on a label the look did not answer is kept as it is.
//!   An answered `tighten` stays and is not asked again: [`refine::same_finding`] keys a `tighten`
//!   on its target box (the exact `bbox` and its label), whatever box it proposed.
//! - **A failure** of the look (the environment, the weights, the memory, the script) keeps the
//!   folds' suggestions: the check ends as usual, the failure in its summary and the report's
//!   warnings.
//!
//! [`refine::same_finding`]: crate::collect::labels::refine::same_finding

use super::job::{Acc, RefineJob};
use super::judge::Loose;
use super::write;
use crate::collect::labels::overlaps::Det as ZoomDet;
use crate::training::autolabel::triage::Det;
use crate::training::overlaps::select::{Crop, Selected};
use crate::training::overlaps::zoom::{self, CropResult};
use crate::training::pyenv::{is_cancel, stage};
use crate::training::task::{TaskContext, TaskReport};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// The candidates of the check, in its work folder: one JSON line per frame.
pub const CANDIDATES_FILE: &str = "tighten-candidates.jsonl";
/// The look's own folder in the work folder (the crop list and the script's answers).
pub const ZOOM_DIR: &str = "zoom";
/// Seconds per crop of the look: the overlap check's measurement (79 ms per crop of RF-DETR L on a
/// GTX 1070 Ti, 3,578 crops in about 5 minutes; its Review page estimate uses 0.08 s).
pub const ZOOM_S_PER_CROP: f64 = 0.079;
/// Seconds of the look beyond its crops: the environment check, the script's self-check and
/// loading RF-DETR L. A rough figure (not measured apart from the crops); the preview says "a few
/// minutes".
pub const ZOOM_FIXED_S: f64 = 90.0;

/// Rough seconds of the look over a check of `frames` frames, before anyone knows how many loose
/// labels it has: one crop per frame at most (on 851 reviewed frames the fold models found 148
/// loose labels, fewer of them confident), plus [`ZOOM_FIXED_S`].
pub fn estimate_seconds(frames: usize) -> f64 {
    #[allow(clippy::cast_precision_loss, reason = "a count of frames")]
    let n = frames as f64;
    ZOOM_FIXED_S + ZOOM_S_PER_CROP * n
}

/// The preview's warning when a check runs without the look because RF-DETR L's weights are not
/// downloaded, with how to get them (the overlap check downloads them the same way).
pub fn skipped_warning() -> String {
    "loose labels are counted but get no better-fitting box: the closer look needs RF-DETR L's \
     weights (0.14 GB, the overlap check's model), which are not downloaded. Tick \"Propose better \
     boxes for loose labels\" (CLI: --tighten) to download them with the check, as the overlap \
     check does (Review tab: Download RF-DETR L if missing)"
        .to_string()
}

/// A fold model's detection as the candidates file keeps it ([`Det`]'s fields).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedDet {
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
}

impl From<&Det> for SavedDet {
    fn from(d: &Det) -> Self {
        Self {
            label: d.label.clone(),
            score: d.score,
            bbox: d.bbox,
        }
    }
}

impl From<&SavedDet> for Det {
    fn from(d: &SavedDet) -> Self {
        Self {
            label: d.label.clone(),
            score: d.score,
            bbox: d.bbox,
        }
    }
}

/// One frame with loose labels to look at closer: what its fold's write judged, and what the
/// write needs to judge it afresh (the fold model's detections, the frame's size and place).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameCandidates {
    /// `<scene>/<camera>/<date>/<stem>.jpg`.
    pub rel: String,
    /// The image the crop is cut from.
    pub image: PathBuf,
    pub width: u32,
    pub height: u32,
    pub camera: String,
    pub day: String,
    /// The fold model's detections on the frame.
    pub dets: Vec<SavedDet>,
    /// The candidates ([`super::judge::Judged::candidates`]).
    pub loose: Vec<Loose>,
}

/// Append `frames` to `<work>/`[`CANDIDATES_FILE`] (created when missing), flushed to disk.
pub fn append(work: &Path, frames: &[FrameCandidates]) -> Result<()> {
    if frames.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(work).with_context(|| format!("creating {}", work.display()))?;
    let path = work.join(CANDIDATES_FILE);
    let mut bytes = Vec::new();
    for f in frames {
        serde_json::to_writer(&mut bytes, f)?;
        bytes.push(b'\n');
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|()| file.flush())
        .and_then(|()| file.sync_data())
        .with_context(|| format!("writing {}", path.display()))
}

/// The candidates of `<work>/`[`CANDIDATES_FILE`]: none when there is no file; a line that does
/// not parse is left out with a warning (its frame gets no closer look).
pub fn read(work: &Path, warnings: &mut Vec<String>) -> Result<Vec<FrameCandidates>> {
    let path = work.join(CANDIDATES_FILE);
    let text = match std::fs::read_to_string(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        t => t.with_context(|| format!("reading {}", path.display()))?,
    };
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match serde_json::from_str::<FrameCandidates>(line) {
            Ok(f) => out.push(f),
            Err(e) => warnings.push(format!(
                "a loose label's record does not parse ({e}); its frame got no closer look"
            )),
        }
    }
    Ok(out)
}

/// The loose labels of `frames`.
pub fn count(frames: &[FrameCandidates]) -> usize {
    frames.iter().map(|f| f.loose.len()).sum()
}

/// The script's input for `frames`: one crop per candidate (its [`Loose::union`]), ids in order;
/// and per id, the (frame, candidate) it answers.
pub fn crops(frames: &[FrameCandidates]) -> (Vec<Selected>, Vec<(usize, usize)>) {
    let mut out = Vec::new();
    let mut ids = Vec::new();
    for (fi, f) in frames.iter().enumerate() {
        let crops: Vec<Crop> = f
            .loose
            .iter()
            .enumerate()
            .map(|(li, l)| {
                let id = ids.len();
                ids.push((fi, li));
                Crop {
                    id,
                    union: l.union(),
                }
            })
            .collect();
        if crops.is_empty() {
            continue;
        }
        out.push(Selected {
            rel: f.rel.clone(),
            jpg: f.image.clone(),
            crops,
            pairs: BTreeMap::new(),
        });
    }
    (out, ids)
}

/// A frame of `frames` with the answers the look gave its candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct Looked<'a> {
    pub frame: &'a FrameCandidates,
    /// The answered candidates with the closer look's detections on their crop.
    pub answers: Vec<(Loose, Vec<ZoomDet>)>,
}

/// The frames of `frames` with at least one answered crop of `results` (by the ids of
/// [`crops`]), and how many crops were answered. An unreadable frame's crop is no answer.
pub fn looked<'a>(
    frames: &'a [FrameCandidates],
    ids: &[(usize, usize)],
    results: &BTreeMap<usize, CropResult>,
) -> (Vec<Looked<'a>>, usize) {
    let mut per: BTreeMap<usize, Vec<(Loose, Vec<ZoomDet>)>> = BTreeMap::new();
    let mut n = 0;
    for (id, r) in results {
        let (Some(&(fi, li)), CropResult::Boxes(dets)) = (ids.get(*id), r) else {
            continue;
        };
        let Some(l) = frames.get(fi).and_then(|f| f.loose.get(li)) else {
            continue;
        };
        n += 1;
        per.entry(fi).or_default().push((l.clone(), dets.clone()));
    }
    let out = per
        .into_iter()
        .filter_map(|(fi, answers)| {
            Some(Looked {
                frame: frames.get(fi)?,
                answers,
            })
        })
        .collect();
    (out, n)
}

/// How the look ended.
#[derive(Debug, Default)]
pub struct ZoomEnd {
    /// Stopped by Cancel (the crops that finished were written).
    pub cancelled: bool,
    /// Why it failed; the folds' suggestions are saved all the same.
    pub failed: Option<String>,
    /// Crops answered and crops in all.
    pub done: usize,
    pub crops: usize,
}

/// "1 loose label", "N loose labels".
fn labels(n: usize) -> String {
    format!("{n} loose label{}", if n == 1 { "" } else { "s" })
}

/// The stage's name for `n` candidates.
pub fn stage_name(n: usize) -> String {
    format!("looking closer at {}", labels(n))
}

/// Where the look's parts go within its one step: setting up (the environment, the script's
/// self-check, the weights) up to [`SETUP_TO`], the crops up to [`CROPS_TO`], then the writes. Each
/// part counts from 0 on its own; squeezed into its band, and never below what was shown, the
/// step's bar never goes back.
const SETUP_TO: u8 = 20;
const CROPS_TO: u8 = 95;

/// The highest percent of the look's step shown so far ([`Bar::within`]).
struct Bar {
    last: u8,
}

impl Bar {
    /// `report` with its stage reports' own percent mapped into `lo..=hi` of the step, never below
    /// what was shown before.
    fn within<'a>(
        &'a mut self,
        report: &'a mut dyn FnMut(TaskReport<'_>),
        (lo, hi): (u8, u8),
    ) -> impl FnMut(TaskReport<'_>) + 'a {
        move |r: TaskReport<'_>| match r {
            TaskReport::Stage {
                stage: name,
                step,
                steps,
                step_percent,
                detail,
                ..
            } => {
                let span = u16::from(hi.saturating_sub(lo));
                let p = lo.saturating_add(
                    u8::try_from(u16::from(step_percent.min(100)) * span / 100).unwrap_or(0),
                );
                self.last = self.last.max(p);
                stage(report, name, step, steps, self.last, detail);
            }
            other => report(other),
        }
    }
}

/// Run the look of `job` on `frames` as step `step` of `steps`, write what it supports and count
/// it into `acc` (`tighten_candidates`, `zoomed`, `found.tighten`, `written.tighten`, the frames
/// changed, the warnings). Never an error: a failure is in [`ZoomEnd::failed`] and the report's
/// warnings, the folds' suggestions untouched.
#[allow(
    clippy::too_many_arguments,
    reason = "the job, its work, the report, the stage, cancel"
)]
pub fn run_stage(
    job: &RefineJob,
    frames: &[FrameCandidates],
    ctx: TaskContext<'_>,
    report: &mut dyn FnMut(TaskReport<'_>),
    (step, steps): (usize, usize),
    cancel: &AtomicBool,
    acc: &mut Acc,
) -> ZoomEnd {
    let n = count(frames);
    acc.rep.tighten_candidates = n;
    let name = stage_name(n);
    let mut end = ZoomEnd {
        crops: n,
        ..ZoomEnd::default()
    };
    if n == 0 {
        stage(report, &name, step, steps, 100, "no loose label to look at");
        return end;
    }
    let helper = job.zoom_helper();
    let work = job.work.join(ZOOM_DIR);
    let (sel, ids) = crops(frames);
    let fail = |acc: &mut Acc, end: &mut ZoomEnd, e: anyhow::Error| {
        let why = format!("{e:#}");
        acc.rep.warnings.push(format!(
            "the closer look at loose labels failed: {why}; the other suggestions were saved"
        ));
        end.failed = Some(why);
    };
    stage(report, &name, step, steps, 0, "setting up the environment");
    let mut bar = Bar { last: 0 };
    let prepared = zoom::prepare(
        &helper,
        job.request.allow_large,
        ctx,
        &mut bar.within(report, (0, SETUP_TO)),
        (&name, step, steps),
        cancel,
    );
    let (script, weights) = match prepared {
        Ok(p) => p,
        Err(e) if is_cancel(&e) => {
            end.cancelled = true;
            return end;
        }
        Err(e) => {
            fail(acc, &mut end, e);
            return end;
        }
    };
    if let Some(why) = zoom::commit_problem("the closer look at loose labels", (job.free_commit)())
    {
        fail(acc, &mut end, anyhow::anyhow!(why));
        return end;
    }
    let ran = match zoom::run(
        &helper,
        &work,
        &script,
        &weights,
        &job.classes,
        &sel,
        n,
        ctx,
        &mut bar.within(report, (SETUP_TO, CROPS_TO)),
        (&name, step, steps),
        cancel,
    ) {
        Ok(r) => r,
        Err(e) => {
            fail(acc, &mut end, e);
            return end;
        }
    };
    end.cancelled = ran.stopped;
    let partial = ran.stopped || ran.failed.is_some();
    let results = match zoom::read_results(&ran.out, partial, &mut acc.rep.warnings) {
        Ok(r) => r,
        Err(e) => {
            fail(acc, &mut end, e);
            return end;
        }
    };
    if let Some(e) = ran.failed {
        fail(acc, &mut end, e);
    }
    let (looked, done) = looked(frames, &ids, &results);
    end.done = done;
    acc.rep.zoomed = done;
    // The writes: no cancel checks, a stopped look writes what it has.
    let wanted = job.wanted();
    let empty = BTreeMap::new();
    let jctx = job.tighten_context(&wanted, &empty);
    let mut writes = bar.within(report, (CROPS_TO, 99));
    let w = write::write_tighten(&job.dataset_dir, &looked, &jctx, &mut |d, total| {
        let pct = d * 100 / total.max(1);
        stage(
            &mut writes,
            &name,
            step,
            steps,
            u8::try_from(pct).unwrap_or(100),
            &format!("writing better-fitting boxes: {d} of {total} frames"),
        );
    });
    drop(writes);
    acc.note_tighten(&w);
    stage(
        report,
        &name,
        step,
        steps,
        100,
        &format!("{done} of {n} looked at, {} better-fitting boxes", w.found),
    );
    end
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn frame(rel: &str, loose: Vec<Loose>) -> FrameCandidates {
        FrameCandidates {
            rel: rel.into(),
            image: PathBuf::from(format!("{rel}.jpg")),
            width: 640,
            height: 480,
            camera: "cam".into(),
            day: "2026-10-01".into(),
            dets: vec![SavedDet {
                label: "car".into(),
                score: 0.9,
                bbox: [1.0, 2.0, 3.0, 4.0],
            }],
            loose,
        }
    }

    fn pair(x: f32) -> Loose {
        Loose {
            target: [x, 10.0, x + 50.0, 60.0],
            target_label: "car".into(),
            class: "car".into(),
            det: [x + 5.0, 5.0, x + 40.0, 70.0],
            det_score: 0.8,
        }
    }

    #[test]
    fn candidates_append_per_fold_and_read_back() {
        let d = std::env::temp_dir().join(format!("bop-tighten-{}", uuid::Uuid::new_v4()));
        let mut warnings = Vec::new();
        assert!(read(&d, &mut warnings).unwrap().is_empty(), "no file: none");
        let a = frame("day/cam/2026-10-01/a.jpg", vec![pair(0.0), pair(100.0)]);
        let b = frame("day/cam/2026-10-02/b.jpg", vec![pair(200.0)]);
        append(&d, std::slice::from_ref(&a)).unwrap();
        append(&d, &[]).unwrap();
        append(&d, std::slice::from_ref(&b)).unwrap();
        // A torn line (a crash mid-write) is left out with a warning.
        let path = d.join(CANDIDATES_FILE);
        let mut t = std::fs::read_to_string(&path).unwrap();
        t.push_str("{\"rel\":\"x\"\n");
        std::fs::write(&path, t).unwrap();
        let back = read(&d, &mut warnings).unwrap();
        assert_eq!(back, vec![a, b]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(count(&back), 3);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn one_crop_per_candidate_on_the_union_and_answers_back_to_their_pair() {
        let frames = vec![
            frame("a", vec![pair(0.0), pair(100.0)]),
            frame("empty", vec![]),
            frame("b", vec![pair(200.0)]),
        ];
        let (sel, ids) = crops(&frames);
        assert_eq!(sel.len(), 2, "a frame without candidates has no crop");
        assert_eq!(ids, [(0, 0), (0, 1), (2, 0)]);
        assert_eq!(sel[0].crops[1].id, 1);
        assert_eq!(sel[0].crops[0].union, [0.0, 5.0, 50.0, 70.0]);
        assert_eq!(sel[1].crops[0].id, 2);
        assert_eq!(sel[1].jpg, PathBuf::from("b.jpg"));
        // The script's input names each crop with its union.
        let text = zoom::crop_list(&sel).unwrap();
        assert_eq!(text.lines().count(), 2);
        assert!(text.contains("\"union\":[0.0,5.0,50.0,70.0]"), "{text}");
        let car = ZoomDet {
            label: "car".into(),
            bbox: [1.0, 1.0, 9.0, 9.0],
            score: 0.7,
        };
        let results = BTreeMap::from([
            (1, CropResult::Boxes(vec![car.clone()])),
            (2, CropResult::Unreadable),
            (7, CropResult::Boxes(vec![])),
        ]);
        let (l, n) = looked(&frames, &ids, &results);
        assert_eq!(n, 1, "an unreadable frame or an unknown id is no answer");
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].frame.rel, "a");
        assert_eq!(l[0].answers, vec![(pair(100.0), vec![car])]);
    }

    #[test]
    fn the_estimate_is_a_few_minutes_for_a_real_dataset() {
        // 851 frames: about 2.6 minutes at most.
        let s = estimate_seconds(851);
        assert!((150.0..200.0).contains(&s), "{s}");
        assert_eq!(stage_name(1), "looking closer at 1 loose label");
        assert_eq!(stage_name(148), "looking closer at 148 loose labels");
        assert!(skipped_warning().contains("--tighten"));
    }
}
