//! Detection with a model of the config in this process (labelling with a model, the
//! self-training loop): the frames go through the benchmark's per-image pipeline
//! ([`crate::benchmark::Bench`]: decode, preprocess, infer, postprocess, the code the workers
//! run), at a given confidence, on the device the model is configured for or the run's device.
//! [`Detector`] is the seam the job uses (tests fake it).

use super::triage::Det;
use crate::backend::{CoreOptions, Runtimes};
use crate::benchmark::images::{GroundTruth, ImageSet, SetImage, SetKind};
use crate::benchmark::{ImagePred, ImageRun};
use crate::config::{Config, ModelConfig};
use crate::model::PostParams;
use crate::sync::LockExt;
use anyhow::{Context, Result, bail};
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

/// What a [`Detector`] found.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Detections {
    /// Detections per frame (`<scene>/<camera>/<date>/<stem>.jpg`).
    pub frames: BTreeMap<String, Vec<Det>>,
    /// Canonical spec of the device that ran (`openvino:cpu`, `ort:coreml`; empty if unknown).
    pub device: String,
    /// The cancel flag stopped the run: `frames` holds the frames finished before it.
    pub cancelled: bool,
}

/// Runs a model of the config over frames.
pub trait Detector: Send + Sync {
    /// Detections of `model` (an entry of `config`, enabled or not) at `threshold` on each of
    /// `images`, on `device` (a device spec; None = the model's configured device). A cancel
    /// ends it with the frames finished so far and [`Detections::cancelled`] set.
    fn detect(
        &self,
        config: &Config,
        model: &ModelConfig,
        device: Option<&str>,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<Detections>;
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
        config: &Config,
        m: &ModelConfig,
        device: Option<&str>,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<Detections> {
        if images.is_empty() {
            return Ok(Detections::default());
        }
        let model = m.effective_name();
        let job = crate::benchmark::job_for_config(config, m, device)?;
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
                nms_iou: config.nms_iou,
            },
        );
        bench.cancel = Some(cancel);
        bench.max_stored_preds = 300;
        // Each frame as it finishes, for a cancelled run.
        let finished: Mutex<BTreeMap<String, Vec<Det>>> = Mutex::default();
        let sink = |r: &ImageRun| {
            finished
                .lock_recover()
                .insert(r.file.clone(), dets_of(&r.preds, threshold));
        };
        bench.on_image = Some(&sink);
        let rt = self.runtimes()?;
        let run = match bench.run(&rt, &job, &job.device) {
            Err(e) if crate::benchmark::is_cancelled(&e) => {
                return Ok(Detections {
                    frames: finished.into_inner().unwrap_or_else(|p| p.into_inner()),
                    device: String::new(),
                    cancelled: true,
                });
            }
            r => r.with_context(|| format!("running {model} on the frames"))?,
        };
        if run.per_image.len() != images.len() {
            bail!(
                "{model} returned results for {} of {} frames",
                run.per_image.len(),
                images.len()
            );
        }
        let device = if run.spec.is_empty() {
            run.device.clone()
        } else {
            run.spec.clone()
        };
        let frames = run
            .per_image
            .into_iter()
            .map(|r| {
                let dets = dets_of(&r.preds, threshold);
                (r.file, dets)
            })
            .collect();
        Ok(Detections {
            frames,
            device,
            cancelled: false,
        })
    }
}

/// The detections of an image's predictions at `threshold` or more.
fn dets_of(preds: &[ImagePred], threshold: f32) -> Vec<Det> {
    preds
        .iter()
        .filter(|p| p.confidence >= threshold)
        .map(|p| Det {
            label: p.label.clone(),
            score: p.confidence,
            bbox: p.bbox,
        })
        .collect()
}
