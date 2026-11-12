//! Stage 3 of the overlap check: `scripts/autolabel.py --zoom-pairs` runs RF-DETR L on the crops of
//! [`super::select`], one flushed JSONL line per finished crop (`@overlaps` events carry the
//! progress). The answers ([`read_results`]) are detections only; every decision is the service's
//! ([`crate::collect::labels::overlaps`]).
//!
//! The dataset check's closer look at loose labels ([`crate::training::refine::tighten`]) runs the
//! same script on its own crops: [`prepare`] (the environment, the script's self-check, RF-DETR L's
//! weights) and [`run`] (the script, its progress, how it ended) serve both.

use super::NEEDED_COMMIT_GB;
use super::select::{Crop, Selected};
use crate::backend::memory::Commit;
use crate::collect::labels::overlaps::{Det, ZOOM_FLOOR, ZOOM_PAD};
use crate::resources::catalog::Platform;
use crate::resources::export::train::{self as env, TrainPaths};
use crate::training::capability::Accelerator;
use crate::training::pyenv::{self, check, is_cancel, stage};
use crate::training::task::{TaskContext, TaskReport};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Event tag of the script in this mode.
pub const TAG: &str = "overlaps";

/// One frame of the script's input.
#[derive(Debug, Serialize)]
struct FrameLine<'a> {
    frame: &'a str,
    path: &'a Path,
    crops: &'a [Crop],
}

/// One crop of the script's output.
#[derive(Debug, Deserialize)]
struct ZoomLine {
    id: usize,
    #[serde(default)]
    boxes: Option<Vec<Det>>,
}

/// What a crop answered.
#[derive(Debug, Clone, PartialEq)]
pub enum CropResult {
    /// Its detections, in frame coordinates.
    Boxes(Vec<Det>),
    /// The frame could not be read: there is no zoom answer.
    Unreadable,
}

/// The script's input: one line per frame with crops.
pub fn crop_list(frames: &[Selected]) -> Result<String> {
    let mut text = String::new();
    for f in frames.iter().filter(|f| !f.crops.is_empty()) {
        text.push_str(&serde_json::to_string(&FrameLine {
            frame: &f.rel,
            path: &f.jpg,
            crops: &f.crops,
        })?);
        text.push('\n');
    }
    Ok(text)
}

/// The script's arguments after its path.
pub fn args(
    list: &Path,
    out: &Path,
    weights: &Path,
    classes: &[String],
    device: &str,
) -> Result<Vec<OsString>> {
    Ok(vec![
        "--zoom-pairs".into(),
        list.as_os_str().to_owned(),
        "--out".into(),
        out.as_os_str().to_owned(),
        "--rfdetr".into(),
        weights.as_os_str().to_owned(),
        "--classes".into(),
        serde_json::to_string(classes)?.into(),
        "--device".into(),
        device.into(),
        "--rfdetr-threshold".into(),
        ZOOM_FLOOR.to_string().into(),
        "--zoom-pad".into(),
        ZOOM_PAD.to_string().into(),
    ])
}

/// Where and how the script runs: the training environment and the GPU of the job.
#[derive(Debug, Clone, Copy)]
pub struct Helper<'a> {
    pub paths: &'a TrainPaths,
    pub platform: Platform,
    pub accelerator: &'a Accelerator,
    pub timeout: Duration,
}

/// Why `free` cannot hold the script (`what` needs it: "the overlap check", ...), or None (enough,
/// or off Windows: no fixed limit). Measured: [`NEEDED_COMMIT_GB`].
pub fn commit_problem(what: &str, free: Option<Commit>) -> Option<String> {
    let needed = NEEDED_COMMIT_GB * 1024.0 * 1024.0 * 1024.0;
    let have = free?.available as f64;
    (have < needed).then(|| {
        format!(
            "{what} needs about {NEEDED_COMMIT_GB} GB of free commit for PyTorch and RF-DETR L, \
             and {:.1} GB is free: close programs or enlarge the paging file",
            have / (1024.0 * 1024.0 * 1024.0)
        )
    })
}

/// The environment, the script's own check and RF-DETR L's weights (downloaded when missing, the
/// large-download gate by `allow_large`), reported as stage `step` of `steps` named `name`; the
/// script and the weights file.
pub fn prepare(
    h: &Helper<'_>,
    allow_large: bool,
    ctx: TaskContext<'_>,
    report: &mut dyn FnMut(TaskReport<'_>),
    (name, step, steps): (&str, usize, usize),
    cancel: &AtomicBool,
) -> Result<(PathBuf, PathBuf)> {
    pyenv::setup(
        h.paths,
        h.platform,
        h.accelerator,
        ctx.runner,
        ctx.fetcher,
        report,
        (name, step, steps),
        cancel,
    )?;
    let (_, script) = env::write_task_scripts(h.paths)?;
    stage(report, name, step, steps, 90, "checking the auto-labeler");
    pyenv::self_check(ctx.runner, h.paths, h.platform, &script, report, cancel)?;
    check(cancel)?;
    let weights = [super::rfdetr_weights(h.paths)?];
    pyenv::check_large(&weights, allow_large)?;
    pyenv::fetch_missing(ctx.fetcher, &weights, report, (name, step, steps), cancel)?;
    let [(res, dir)] = weights;
    let file = res
        .parts
        .first()
        .ok_or_else(|| anyhow!("RF-DETR L has no file"))?
        .file_name;
    Ok((script, dir.join(file)))
}

/// How a [`run`] ended. The finished crops are in `out` whichever way.
#[derive(Debug)]
pub struct Ran {
    /// Stopped by Cancel.
    pub stopped: bool,
    /// The failure that stopped it.
    pub failed: Option<anyhow::Error>,
    /// The script's output (`zoom.jsonl` in the work folder).
    pub out: PathBuf,
}

/// Run the script on the crops of `frames` (`total` of them) in `work` (`crops.jsonl`, `zoom.jsonl`),
/// RF-DETR L at `weights` seeing `classes`, reporting stage `step` of `steps` named `name` ("RF-DETR
/// L loaded", "i of N crops"). An error only when the work folder or the crop list cannot be
/// written; the script's own end is in [`Ran`].
#[allow(
    clippy::too_many_arguments,
    reason = "the helper, the work, the model and its classes, the report and its stage, cancel"
)]
pub fn run(
    h: &Helper<'_>,
    work: &Path,
    script: &Path,
    weights: &Path,
    classes: &[String],
    frames: &[Selected],
    total: usize,
    ctx: TaskContext<'_>,
    report: &mut dyn FnMut(TaskReport<'_>),
    (name, step, steps): (&str, usize, usize),
    cancel: &AtomicBool,
) -> Result<Ran> {
    std::fs::create_dir_all(work).with_context(|| format!("creating {}", work.display()))?;
    let list = work.join("crops.jsonl");
    std::fs::write(&list, crop_list(frames)?)
        .with_context(|| format!("writing {}", list.display()))?;
    let out = work.join("zoom.jsonl");
    let mut cmd = vec![script.as_os_str().to_owned()];
    cmd.extend(args(
        &list,
        &out,
        weights,
        classes,
        &h.accelerator.device_arg(),
    )?);
    let mut done_ev = false;
    let ran = pyenv::run_script(
        ctx.runner,
        h.paths,
        h.platform,
        work,
        h.timeout,
        cmd,
        TAG,
        &mut |ev, report| match ev.get("event").and_then(serde_json::Value::as_str) {
            Some("crop") => {
                let i = ev
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let pct =
                    u8::try_from(i * 100 / u64::try_from(total.max(1)).unwrap_or(1)).unwrap_or(100);
                stage(
                    report,
                    name,
                    step,
                    steps,
                    pct.min(99),
                    &format!("{i} of {total} crops"),
                );
            }
            Some("model") => stage(report, name, step, steps, 0, "RF-DETR L loaded"),
            Some("done") => done_ev = true,
            _ => {}
        },
        report,
        cancel,
    );
    let (stopped, failed) = match ran {
        Err(e) if is_cancel(&e) => (true, None),
        Err(e) => (false, Some(e.context("the zoom check failed"))),
        Ok(()) if cancel.load(Ordering::Relaxed) => (true, None),
        Ok(()) if !done_ev => (
            false,
            Some(anyhow!(
                "the zoom check finished without its summary; the script is damaged"
            )),
        ),
        Ok(()) => (false, None),
    };
    Ok(Ran {
        stopped,
        failed,
        out,
    })
}

/// The finished crops of `path` by id. `partial`: the run stopped (a cancel, a failure), so the file
/// may be missing or end in a line cut short; that line is left out. Any other line that does not
/// parse is a warning: its crop has no answer, and a rerun does it again.
pub fn read_results(
    path: &Path,
    partial: bool,
    warnings: &mut Vec<String>,
) -> Result<BTreeMap<usize, CropResult>> {
    let text = match std::fs::read_to_string(path) {
        Err(e) if partial && e.kind() == std::io::ErrorKind::NotFound => String::new(),
        t => t.with_context(|| format!("reading {}", path.display()))?,
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut out = BTreeMap::new();
    for (i, line) in lines.iter().enumerate() {
        match serde_json::from_str::<ZoomLine>(line) {
            Ok(z) => {
                out.insert(
                    z.id,
                    z.boxes.map_or(CropResult::Unreadable, CropResult::Boxes),
                );
            }
            Err(_) if partial && i + 1 == lines.len() => {}
            Err(e) => warnings.push(format!("a zoom result line does not parse: {e}")),
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn results_read_what_finished_and_a_cut_line_is_no_error() {
        let d = std::env::temp_dir().join(format!("bop-zoom-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("zoom.jsonl");
        let mut warnings = Vec::new();
        assert!(read_results(&p, true, &mut warnings).unwrap().is_empty());
        assert!(read_results(&p, false, &mut warnings).is_err(), "no file");
        std::fs::write(
            &p,
            "{\"id\":0,\"frame\":\"f\",\"boxes\":[{\"label\":\"car\",\"bbox\":[0,0,9,9],\"score\":0.7}]}\n\
             {\"id\":1,\"frame\":\"f\",\"error\":\"broken\"}\n\
             {\"id\":2,\"fr",
        )
        .unwrap();
        let got = read_results(&p, true, &mut warnings).unwrap();
        assert_eq!(got.len(), 2);
        assert!(matches!(&got[&0], CropResult::Boxes(b) if b.len() == 1 && b[0].score == 0.7));
        assert_eq!(got[&1], CropResult::Unreadable);
        assert!(warnings.is_empty());
        // A bad line in a finished run: a warning.
        read_results(&p, false, &mut warnings).unwrap();
        assert_eq!(warnings.len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }
}
