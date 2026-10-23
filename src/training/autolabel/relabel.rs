//! Detection with a configured model in this process, for the self-training loop: the frames go
//! through the benchmark's per-image pipeline ([`crate::benchmark::Bench`]: decode, preprocess,
//! infer, postprocess, the code the workers run), at a given confidence, on the device the model
//! is configured for. [`Detector`] is the seam the job uses (tests fake it).

use super::triage::Det;
use crate::backend::{CoreOptions, Runtimes};
use crate::benchmark::images::{GroundTruth, ImageSet, SetImage, SetKind};
use crate::config::Config;
use crate::model::PostParams;
use anyhow::{Context, Result, anyhow, bail};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// A frame to detect on.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    /// `<scene>/<camera>/<date>/<stem>.jpg`.
    pub frame: String,
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
}

/// Runs a configured model over frames.
pub trait Detector: Send + Sync {
    /// Detections of `model` (a configured model's name) at `threshold` on each of `images`,
    /// keyed by frame. Fails on cancel.
    fn detect(
        &self,
        model: &str,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<BTreeMap<String, Vec<Det>>>;
}

/// The real [`Detector`]: the registry's runtimes when given (the service), else its own from
/// the config (the CLI).
pub struct LocalDetector {
    pub config: Config,
    pub runtimes: Option<Arc<Mutex<Runtimes>>>,
}

impl std::fmt::Debug for LocalDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalDetector")
            .field("shared_runtimes", &self.runtimes.is_some())
            .finish_non_exhaustive()
    }
}

impl LocalDetector {
    fn runtimes(&self) -> Result<Arc<Mutex<Runtimes>>> {
        if let Some(rt) = &self.runtimes {
            return Ok(rt.clone());
        }
        let c = &self.config;
        crate::backend::libs::prepare_environment(c.openvino_dir_effective().as_deref());
        let rt = Runtimes::new_with(
            &CoreOptions {
                cache_dir: c.cache_dir_path(),
                intra_threads: c.intra_threads,
                openvino_dir: c.openvino_dir_effective(),
            },
            &c.ort_options(),
        );
        rt.require_any()?;
        Ok(Arc::new(Mutex::new(rt)))
    }
}

impl Detector for LocalDetector {
    fn detect(
        &self,
        model: &str,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<BTreeMap<String, Vec<Det>>> {
        if images.is_empty() {
            return Ok(BTreeMap::new());
        }
        let m = self
            .config
            .models
            .iter()
            .find(|m| m.effective_name().eq_ignore_ascii_case(model))
            .ok_or_else(|| anyhow!("no configured model named {model}"))?;
        let job = crate::benchmark::job_for_config(&self.config, m, None)?;
        let set = ImageSet {
            id: "autolabel".into(),
            title: "frames to re-label".into(),
            kind: SetKind::Dir,
            dir: None,
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: GroundTruth::None,
            scored_labels: None,
            transform: None,
            images: images
                .iter()
                .map(|i| SetImage {
                    file: i.frame.clone(),
                    path: Some(i.path.clone()),
                    width: i.width,
                    height: i.height,
                    tags: Vec::new(),
                    gt: None,
                })
                .collect(),
        };
        let sets = [set];
        let mut bench = crate::benchmark::Bench::new(
            &sets,
            0,
            1,
            PostParams {
                confidence_threshold: threshold,
                nms_iou: self.config.nms_iou,
            },
        );
        bench.cancel = Some(cancel);
        bench.max_stored_preds = 300;
        let rt = self.runtimes()?;
        let run = bench
            .run(&rt, &job, &job.device)
            .with_context(|| format!("running {model} on the frames"))?;
        if run.per_image.len() != images.len() {
            bail!(
                "{model} returned results for {} of {} frames",
                run.per_image.len(),
                images.len()
            );
        }
        Ok(run
            .per_image
            .into_iter()
            .map(|r| {
                let dets = r
                    .preds
                    .into_iter()
                    .filter(|p| p.confidence >= threshold)
                    .map(|p| Det {
                        label: p.label,
                        score: p.confidence,
                        bbox: p.bbox,
                    })
                    .collect();
                (r.file, dets)
            })
            .collect())
    }
}
