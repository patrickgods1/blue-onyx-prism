//! Running a fold model over its held-out frames, in this process ([`FoldPredictor`], the seam the
//! job's tests fake). The real one ([`LocalPredictor`]) goes through the auto-label's
//! [`LocalDetector`] (the benchmark's per-image pipeline: the code the workers run) with a model
//! entry built for the fold's ONNX and class file ([`model_entry`]) on a copy of the config that is
//! never saved: a fold model never appears in the config, the Config page or `/v1/vision/custom`.

use crate::backend::{CoreOptions, Runtimes};
use crate::config::{Config, ModelConfig};
use crate::model::ModelFamilyKind;
use crate::sync::LockExt;
use crate::training::autolabel::relabel::{Detections, Detector, Image, LocalDetector};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// A trained fold model (from [`crate::training::job::TrainOutput`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FoldModel {
    /// `refine-fold-<k>`.
    pub name: String,
    pub onnx: PathBuf,
    /// Its class file (`<name>.yaml` beside the ONNX).
    pub yaml: PathBuf,
    pub family: ModelFamilyKind,
    pub classes: Vec<String>,
}

/// Runs a fold model over frames.
pub trait FoldPredictor: Send + Sync {
    /// Detections of `model` at `threshold` or more on each of `images` (keyed by
    /// [`Image::frame`]). A cancel ends it with the frames finished so far and
    /// [`Detections::cancelled`] set.
    fn predict(
        &self,
        model: &FoldModel,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<Detections>;
}

/// The config entry of `m`: disabled, its own class file, the default device. Used on a copy of the
/// config only, never saved.
pub fn model_entry(m: &FoldModel) -> ModelConfig {
    ModelConfig {
        name: Some(m.name.clone()),
        path: m.onnx.clone(),
        family: m.family,
        classes: Some(m.yaml.clone()),
        enabled: false,
        ..ModelConfig::default()
    }
}

/// The real [`FoldPredictor`]: the registry's runtimes when given (the service: no second
/// OpenVINO / ONNX Runtime instance), else its own, made once for the whole check (the CLI).
pub struct LocalPredictor {
    config: Config,
    runtimes: Mutex<Option<Arc<Mutex<Runtimes>>>>,
}

impl std::fmt::Debug for LocalPredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalPredictor").finish_non_exhaustive()
    }
}

impl LocalPredictor {
    /// A predictor on `config` (a copy: nothing is saved), sharing `runtimes` when given.
    pub fn new(config: Config, runtimes: Option<Arc<Mutex<Runtimes>>>) -> Self {
        Self {
            config,
            runtimes: Mutex::new(runtimes),
        }
    }

    fn runtimes(&self) -> Result<Arc<Mutex<Runtimes>>> {
        let mut slot = self.runtimes.lock_recover();
        if let Some(rt) = slot.as_ref() {
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
        let rt = Arc::new(Mutex::new(rt));
        *slot = Some(rt.clone());
        Ok(rt)
    }
}

impl FoldPredictor for LocalPredictor {
    fn predict(
        &self,
        model: &FoldModel,
        images: &[Image],
        threshold: f32,
        cancel: &AtomicBool,
    ) -> Result<Detections> {
        let detector = LocalDetector {
            config: self.config.clone(),
            runtimes: Some(self.runtimes()?),
        };
        detector.detect(
            &self.config,
            &model_entry(model),
            None,
            images,
            threshold,
            cancel,
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn the_entry_is_disabled_and_names_the_fold_files() {
        let m = FoldModel {
            name: "refine-fold-2".into(),
            onnx: PathBuf::from("/w/fold-2/model/refine-fold-2.onnx"),
            yaml: PathBuf::from("/w/fold-2/model/refine-fold-2.yaml"),
            family: ModelFamilyKind::RfDetr,
            classes: vec!["person".into()],
        };
        let e = model_entry(&m);
        assert!(!e.enabled);
        assert_eq!(e.effective_name(), "refine-fold-2");
        assert_eq!(e.path, m.onnx);
        assert_eq!(e.classes.as_deref(), Some(m.yaml.as_path()));
        assert_eq!(e.family, ModelFamilyKind::RfDetr);
        // Nothing else of the default entry changes: the config's device, thresholds, filters.
        let d = ModelConfig::default();
        assert_eq!(e.device, d.device);
        assert_eq!(e.confidence_threshold, d.confidence_threshold);
    }
}
