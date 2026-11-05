//! The overlap check (`blue-onyx-prism train overlaps`, the Review page's *Suggest overlaps...*;
//! docs/TRAINING.md "Overlap suggestions"): a second look at the nested and partly overlapping boxes
//! of the folder under review. RF-DETR L runs on a crop of each open pair and
//! [`crate::collect::labels::overlaps`] turns what it saw, with the witnesses already in the label
//! file, into suggestions. **It only writes `overlap_suggestions`**: no box state, frame status,
//! `reviewed_at` or `overlaps_checked` changes. A box changes when a reviewer accepts a suggestion
//! on the page, because a blind test found 13 of 99 automatic drops removed a real object.
//!
//! It runs on the training queue like an auto-label run, as its own [`job::OverlapsJob`] task: one job
//! at a time, the training lock, low priority, pause and cancel. Reviewed and unreviewed frames are in
//! scope (382 of the 1,169 frames with overlaps measured were reviewed); skipped frames and frames a
//! reviewer checked (`overlaps_checked`) are not. Each stage is a file:
//!
//! - [`select`]: stage 1, the open pairs without an answer and the crops they need;
//! - [`job`]: stage 2 (the environment, RF-DETR L's weights) and the stage machine;
//! - [`zoom`]: stage 3, `scripts/autolabel.py --zoom-pairs`, one flushed JSONL line per finished crop;
//! - [`write`]: stage 4, the suggestions, re-judged under the label write lock.
//!
//! A cancel keeps the finished crops: they are written as usual, and a rerun skips every pair with an
//! answer, so it resumes.

pub mod job;
pub mod select;
pub mod write;
pub mod zoom;

pub use job::OverlapsJob;

use super::autolabel::{FrameFilter, MAX_LIMIT};
use super::capability::Capability;
use super::pyenv;
use crate::config::Config;
use crate::resources::catalog::{self as res_catalog, Platform, Resource, TrainFamily};
use crate::resources::export::train::{self as env, TrainPaths};
use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Free commit (RAM + paging file) the check needs on Windows, in GiB: torch, RF-DETR L and its
/// trace, measured. A fixed paging file runs out of commit long before RAM.
pub const NEEDED_COMMIT_GB: f64 = 2.6;

/// What to check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlapsRequest {
    /// Which frames: the filters of `train autolabel`. `status` and `source` do not narrow it (every
    /// label file of the folder but skipped and checked frames), `again=1` also looks at pairs with
    /// an answer, `limit` counts frames with work.
    pub frames: FrameFilter,
    /// GPU (`cuda:N` of [`Capability::cuda_options`]; None = the automatic pick).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default)]
    pub allow_large: bool,
}

impl OverlapsRequest {
    /// A request for the frames of `filter` (`scene=ir,camera=patio,limit=50,again=1`).
    pub fn parse(filter: &str) -> Result<Self> {
        let mut frames = FrameFilter::parse(filter, "any")?;
        frames.query.insert("status".into(), "any".into());
        frames.query.insert("source".into(), "all".into());
        if !filter.contains("limit=") {
            frames.limit = MAX_LIMIT;
        }
        Ok(Self {
            frames,
            gpu: None,
            allow_large: false,
        })
    }
}

/// RF-DETR L's training base pin and where it is installed.
fn rfdetr_weights(paths: &TrainPaths) -> Result<(&'static Resource, PathBuf)> {
    let res = res_catalog::base_weights(TrainFamily::Rfdetr, "large")
        .ok_or_else(|| anyhow!("no RF-DETR L pin in this build"))?;
    Ok((res, pyenv::weights_dir(paths, res)))
}

/// Plan a check of `request` on `config` (fails fast on a bad filter, a missing GPU or a large
/// download that was not allowed). It reads the folder under review, like the Review page.
pub fn plan_overlaps(
    config: &Config,
    mut request: OverlapsRequest,
    cap: &Capability,
) -> Result<OverlapsJob> {
    request.frames.filters()?;
    let paths = TrainPaths::for_config(config);
    let platform = Platform::current();
    let allow_large = request.allow_large || config.allow_large_downloads;
    let Some(accelerator) = cap.resolve_gpu(request.gpu.as_deref())? else {
        bail!(
            "the overlap check runs RF-DETR L in the training environment, which needs an NVIDIA \
             GPU or Apple silicon: {}",
            cap.summary()
        );
    };
    if !env::train_env_ready(&paths, platform, accelerator.lane()) && !allow_large {
        let (dl, _) = accelerator.lane().estimate();
        bail!(
            "the overlap check runs in the training environment (about {} download); rerun with \
             --allow-large (Review page: tick the download confirmation)",
            res_catalog::format_size(dl)
        );
    }
    let weights = [rfdetr_weights(&paths)?];
    pyenv::check_large(&weights, allow_large)?;
    request.allow_large = allow_large;
    let id = super::service::new_run_id();
    Ok(OverlapsJob {
        work: super::training_dir(config).join("overlaps").join(&id),
        id,
        request,
        dataset_dir: config.collect.resolved_review_dir(&config.data_root()),
        classes: config.collect.classes.clone(),
        hidden: config.collect.hidden_classes.clone(),
        paths,
        platform,
        accelerator,
        timeout: Duration::from_secs(u64::from(config.training.timeout_hours) * 3600),
        free_commit: crate::system_info::commit,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn a_request_looks_at_every_frame_of_the_folder() {
        let r = OverlapsRequest::parse("").unwrap();
        assert_eq!(r.frames.limit, MAX_LIMIT);
        assert_eq!(r.frames.query["status"], "any");
        assert_eq!(r.frames.query["source"], "all");
        // A status or source in the filter does not narrow it; a limit and `again` hold.
        let r = OverlapsRequest::parse("status=reviewed,source=queue,scene=ir,limit=7,again=1")
            .unwrap();
        assert_eq!(r.frames.limit, 7);
        assert_eq!(r.frames.query["status"], "any");
        assert_eq!(r.frames.query["source"], "all");
        assert!(r.frames.again());
        assert!(OverlapsRequest::parse("scene=dusk").is_err());
    }
}
