//! Detection with a model of the config in this process (labelling with a model, the
//! self-training loop): the frames go through the benchmark's per-image pipeline
//! ([`crate::benchmark::Bench`]: decode, preprocess, infer, postprocess, the code the workers
//! run), at a given confidence, on the device the model is configured for or the run's device.
//! [`Detector`] is the seam the job uses (tests fake it). A job the service queues holds a
//! [`CurrentDetector`], which asks for the current registry generation's runtimes only when it
//! detects ([`RuntimesResolver`]).

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
    /// Seconds before the first frame: the runtime, and the model's read, compile and warm-up
    /// (the benchmark's `compile_ms` + `warmup_ms`). None when not measured (a cancelled run, a
    /// fake).
    pub load_s: Option<f64>,
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

/// Where a queued job finds the runtimes when it runs (the service: the current registry
/// generation's, read through its `Shared`), so a job queued before a generation restart neither
/// keeps the retired generation's OpenVINO Core and ONNX Runtime alive while it waits nor
/// compiles on them (audit 2026-11, RA-100). Called once per detection or prediction; the answer
/// is dropped when that returns. None: no runtimes to share (the detector makes its own).
pub type RuntimesResolver = Arc<dyn Fn() -> Option<Arc<Mutex<Runtimes>>> + Send + Sync>;

/// The real [`Detector`]: the registry's runtimes when given (the service), else its own from
/// the config (the CLI). A queued job uses [`CurrentDetector`] instead, which asks for the
/// runtimes only when it runs.
pub struct LocalDetector {
    pub config: Config,
    pub runtimes: Option<Arc<Mutex<Runtimes>>>,
    /// Compile the model without the OpenVINO model cache: a throwaway model (the dataset
    /// check's fold models, [`crate::training::refine::predict`]); false for everything else.
    pub no_cache: bool,
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
        own_runtimes(&self.config)
    }
}

/// Runtimes of this process's own, from `c` (the CLI, or nothing to share).
pub(crate) fn own_runtimes(c: &Config) -> Result<Arc<Mutex<Runtimes>>> {
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

/// A [`LocalDetector`] whose runtimes are resolved when it detects ([`RuntimesResolver`]), not
/// when its job is queued: what a queued job of the service holds.
pub struct CurrentDetector {
    pub config: Config,
    pub resolve: RuntimesResolver,
    /// [`LocalDetector::no_cache`].
    pub no_cache: bool,
}

impl std::fmt::Debug for CurrentDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CurrentDetector").finish_non_exhaustive()
    }
}

impl Detector for CurrentDetector {
    fn detect(
        &self,
        config: &Config,
        m: &ModelConfig,
        device: Option<&str>,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<Detections> {
        // Held for this detection only (the detector below is dropped with it).
        let local = LocalDetector {
            config: self.config.clone(),
            runtimes: (self.resolve)(),
            no_cache: self.no_cache,
        };
        local.detect(config, m, device, images, threshold, cancel)
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
        let t0 = std::time::Instant::now();
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
        bench.no_cache = self.no_cache;
        // Only the detections are kept: read the next frame ahead on every device, CPU too.
        bench.read_ahead = crate::benchmark::ReadAhead::Always;
        // Each frame as it finishes, for a cancelled run.
        let finished: Mutex<BTreeMap<String, Vec<Det>>> = Mutex::default();
        let sink = |r: &ImageRun| {
            finished
                .lock_recover()
                .insert(r.file.clone(), dets_of(&r.preds, threshold));
        };
        bench.on_image = Some(&sink);
        let rt = self.runtimes()?;
        let before_run = t0.elapsed().as_secs_f64();
        let run = match bench.run(&rt, &job, &job.device) {
            Err(e) if crate::benchmark::is_cancelled(&e) => {
                return Ok(Detections {
                    frames: finished.into_inner().unwrap_or_else(|p| p.into_inner()),
                    device: String::new(),
                    cancelled: true,
                    load_s: None,
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
        // Everything before the first frame: the runtime above, then the benchmark's own
        // measure of the model's load (read, compile, request) and warm-up.
        let load_s = before_run + (run.compile_ms + run.warmup_ms) / 1000.0;
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
            load_s: Some(load_s),
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A queued job's detector asks for the runtimes when it detects, once per detection, and
    /// keeps none of them afterwards (audit 2026-11, RA-100).
    #[test]
    fn the_current_detector_resolves_its_runtimes_per_detection() {
        let rt = Arc::new(Mutex::new(Runtimes::fake(
            crate::backend::fake::FakeScript::new(1, (64, 64)),
        )));
        let asked = Arc::new(AtomicUsize::new(0));
        let (r, n) = (rt.clone(), asked.clone());
        let d = CurrentDetector {
            config: Config::default(),
            resolve: Arc::new(move || {
                n.fetch_add(1, Ordering::Relaxed);
                Some(r.clone())
            }),
            no_cache: false,
        };
        assert_eq!(asked.load(Ordering::Relaxed), 0);
        let m = ModelConfig {
            path: PathBuf::from("/nonexistent/bop-model.onnx"),
            ..ModelConfig::default()
        };
        let image = Image {
            frame: "day/cam/2026-11-23/1.jpg".into(),
            path: PathBuf::from("/nonexistent/1.jpg"),
            width: 8,
            height: 8,
        };
        let cancel = AtomicBool::new(false);
        let config = Config::default();
        for i in 1..=2 {
            let e = d
                .detect(
                    &config,
                    &m,
                    None,
                    std::slice::from_ref(&image),
                    0.5,
                    &cancel,
                )
                .unwrap_err();
            assert!(format!("{e:#}").contains("model file not found"), "{e:#}");
            assert_eq!(asked.load(Ordering::Relaxed), i);
            // The resolver's clone and the test's: the detection kept none.
            assert_eq!(Arc::strong_count(&rt), 2);
        }
    }
}
