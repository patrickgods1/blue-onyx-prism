//! Running a fold model over its held-out frames, in this process ([`FoldPredictor`], the seam the
//! job's tests fake). The real one ([`LocalPredictor`]) goes through the auto-label's
//! [`LocalDetector`] (the benchmark's per-image pipeline: the code the workers run) with a model
//! entry built for the fold's ONNX and class file ([`model_entry`]) on a copy of the config that is
//! never saved: a fold model never appears in the config, the Config page or `/v1/vision/custom`.
//! A frame the pipeline cannot read (deleted since the selection, a damaged JPEG) is left out of
//! the answer instead of failing the fold ([`skipping_unreadable`]); the benchmark and auto-label
//! paths are unchanged. A quick check ([`super::quick`]) runs its configured model the same way,
//! with that model's own entry ([`FoldModel::entry`]) instead of a built one.

use crate::backend::{CoreOptions, Runtimes};
use crate::benchmark::images::SetImage;
use crate::config::{Config, ModelConfig};
use crate::model::ModelFamilyKind;
use crate::sync::LockExt;
use crate::training::autolabel::relabel::{Detections, Detector, Image, LocalDetector};
use anyhow::{Context as _, Result};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// A trained fold model (from [`crate::training::job::TrainOutput`]), or the model of a quick
/// check ([`super::quick`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FoldModel {
    /// `refine-fold-<k>`; a quick check's model's configured name.
    pub name: String,
    pub onnx: PathBuf,
    /// Its class file (`<name>.yaml` beside the ONNX).
    pub yaml: PathBuf,
    pub family: ModelFamilyKind,
    pub classes: Vec<String>,
    /// A quick check: the model's config entry as it is configured (its own family, class file and
    /// device; [`super::quick::predict_entry`] took off what would hide detections), used instead
    /// of [`model_entry`]'s. None for a fold model.
    pub entry: Option<ModelConfig>,
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

/// The config entry of `m`: disabled, its own class file, the default device (a quick check's
/// model: [`FoldModel::entry`]). Used on a copy of the config only, never saved.
pub fn model_entry(m: &FoldModel) -> ModelConfig {
    if let Some(e) = &m.entry {
        return e.clone();
    }
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
        let entry = model_entry(model);
        skipping_unreadable(
            images,
            &mut |imgs| detector.detect(&self.config, &entry, None, imgs, threshold, cancel),
            &readable,
        )
    }
}

/// Whether the detection pipeline can read `img`: its bytes as the benchmark reads them
/// ([`SetImage::bytes`]: gone, unreadable or too large fails) through the decoder the workers use
/// ([`crate::image::decode`]: a damaged JPEG fails).
pub fn readable(img: &Image) -> Result<()> {
    let set = SetImage {
        file: img.frame.clone(),
        path: Some(img.path.clone()),
        width: img.width,
        height: img.height,
        tags: Vec::new(),
        gt: None,
    };
    let bytes = set.bytes()?;
    crate::image::decode(&bytes).map(|_| ())
}

/// Most unreadable frames named in the log line of [`skipping_unreadable`].
const NAMED: usize = 5;

/// `detect` over `images`, surviving frames the pipeline cannot read: the detector runs every
/// image in one pass and fails it whole on one bad image, so after a failure (not a cancel) every
/// image is checked with `readable`, those that fail are left out of the answer (the job's writer
/// passes over a frame without a result, with a warning, never judging it as "nothing found") and
/// the rest run once more. A failure with every image readable, or a second failure, is the
/// error. The good case costs nothing extra.
pub fn skipping_unreadable(
    images: &[Image],
    detect: &mut dyn FnMut(&[Image]) -> Result<Detections>,
    readable: &dyn Fn(&Image) -> Result<()>,
) -> Result<Detections> {
    let first = match detect(images) {
        Ok(d) => return Ok(d),
        Err(e) => e,
    };
    let mut bad: Vec<String> = Vec::new();
    let good: Vec<Image> = images
        .iter()
        .filter(|i| match readable(i) {
            Ok(()) => true,
            Err(e) => {
                bad.push(format!("{}: {e:#}", i.frame));
                false
            }
        })
        .cloned()
        .collect();
    if bad.is_empty() {
        return Err(first);
    }
    tracing::warn!(
        "dataset check: {} of {} frames cannot be read and are not checked ({}{})",
        bad.len(),
        images.len(),
        bad.iter()
            .take(NAMED)
            .cloned()
            .collect::<Vec<_>>()
            .join("; "),
        if bad.len() > NAMED { "; ..." } else { "" }
    );
    if good.is_empty() {
        return Ok(Detections::default());
    }
    detect(&good).with_context(|| {
        format!(
            "running again without the {} frame(s) that cannot be read (first failure: {first:#})",
            bad.len()
        )
    })
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
            entry: None,
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
        // A quick check's model: its own entry, as given.
        let own = ModelConfig {
            name: Some("mine".into()),
            path: PathBuf::from("models/mine.onnx"),
            device: Some("openvino:gpu.0".into()),
            ..ModelConfig::default()
        };
        let q = FoldModel {
            entry: Some(own.clone()),
            ..m
        };
        assert_eq!(model_entry(&q), own);
    }

    fn image(frame: &str, path: PathBuf) -> Image {
        Image {
            frame: frame.into(),
            path,
            width: 8,
            height: 8,
        }
    }

    #[test]
    fn an_unreadable_frame_is_left_out_and_the_others_are_answered() {
        let dir = std::env::temp_dir().join(format!("bop-refine-predict-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.jpg");
        ::image::RgbImage::new(8, 8).save(&good).unwrap();
        let broken = dir.join("broken.jpg");
        std::fs::write(&broken, [0xFF, 0xD8, 0xFF, 0x00, 1, 2, 3]).unwrap();
        let images = [
            image("ir/a/2026-10-21/1.jpg", good.clone()),
            image("ir/a/2026-10-21/2.jpg", broken.clone()),
            image("ir/a/2026-10-21/3.jpg", dir.join("gone.jpg")),
            image("ir/a/2026-10-21/4.jpg", good.clone()),
        ];
        readable(&images[0]).unwrap();
        assert!(readable(&images[1]).is_err(), "a damaged JPEG");
        assert!(
            readable(&images[2]).is_err(),
            "a frame deleted since the selection"
        );
        // A detector like the real one: one pass, failed whole by one image it cannot read.
        let calls = std::cell::RefCell::new(Vec::new());
        let mut detect = |imgs: &[Image]| -> Result<Detections> {
            calls
                .borrow_mut()
                .push(imgs.iter().map(|i| i.frame.clone()).collect::<Vec<_>>());
            for i in imgs {
                readable(i).with_context(|| format!("image {}", i.frame))?;
            }
            Ok(Detections {
                frames: imgs.iter().map(|i| (i.frame.clone(), Vec::new())).collect(),
                device: "ort:cpu".into(),
                cancelled: false,
            })
        };
        let d = skipping_unreadable(&images, &mut detect, &readable).unwrap();
        let answered: Vec<&str> = d.frames.keys().map(String::as_str).collect();
        assert_eq!(answered, ["ir/a/2026-10-21/1.jpg", "ir/a/2026-10-21/4.jpg"]);
        assert!(!d.cancelled);
        assert_eq!(
            calls.borrow().len(),
            2,
            "the batch, then the readable frames once"
        );
        // All readable: one pass. A failure with every frame readable is the error.
        calls.borrow_mut().clear();
        let ok = [images[0].clone(), images[3].clone()];
        assert_eq!(
            skipping_unreadable(&ok, &mut detect, &readable)
                .unwrap()
                .frames
                .len(),
            2
        );
        assert_eq!(calls.borrow().len(), 1);
        let mut broken_model = |_: &[Image]| -> Result<Detections> { anyhow::bail!("no device") };
        let e = skipping_unreadable(&ok, &mut broken_model, &readable).unwrap_err();
        assert_eq!(e.to_string(), "no device");
        // Nothing readable: no answer, no error (every frame gets the writer's warning).
        let none = [images[1].clone(), images[2].clone()];
        assert!(
            skipping_unreadable(&none, &mut detect, &readable)
                .unwrap()
                .frames
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
