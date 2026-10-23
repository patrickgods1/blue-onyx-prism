//! The `training` config section: where runs go, which review exports the Training page offers,
//! and the dataset assembly defaults. Read when a run is queued (every field applies at once);
//! pure data plus validation.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Folder name of the default training folder under the data root.
pub const DEFAULT_DIR_NAME: &str = "training";
/// Default epochs of a run.
pub const DEFAULT_EPOCHS: u32 = 50;
/// Most epochs of a run.
pub const MAX_EPOCHS: u32 = 1000;
/// Default share of frames without objects in the training split.
pub const DEFAULT_NEGATIVE_SHARE: f32 = 0.15;
/// Default share of the review frames held out (by camera + day) for the benchmark.
pub const DEFAULT_HELDOUT_SHARE: f32 = 0.2;
/// Default repeat factor of the review frames when public frames are mixed in.
pub const DEFAULT_OVERSAMPLE: u32 = 3;
/// Most repeats.
pub const MAX_OVERSAMPLE: u32 = 10;
/// Default longest run before it is stopped.
pub const DEFAULT_TIMEOUT_HOURS: u32 = 48;
/// Most hours of a run.
pub const MAX_TIMEOUT_HOURS: u32 = 24 * 14;

/// `training` in the config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// Folder of the training runs (prepared datasets, logs, held-out sets). None =
    /// `<data root>/training`; relative paths resolve against the data root.
    pub dir: Option<PathBuf>,
    /// Review export folders (YOLO layout, see docs/TRAINING.md) offered on the Training page
    /// and used by `train run` without `--dataset`.
    pub datasets: Vec<PathBuf>,
    /// Default epochs of a run.
    pub epochs: u32,
    /// Share of frames without objects (hard negatives) kept in the training split (0-0.9).
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub negative_share: f32,
    /// Share of the review frames held out by camera + day for the benchmark (0-0.9).
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub heldout_share: f32,
    /// How often a review frame repeats in the training split when public frames are mixed in.
    pub oversample: u32,
    /// A run is stopped after this many hours.
    pub timeout_hours: u32,
    /// Auto-label prompt phrases per class (`{"raccoon": ["raccoon", "trash panda"]}`), replacing
    /// the built-in ones of that class (MM-Grounding-DINO prompts, SigLIP species prompts).
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub synonyms: std::collections::BTreeMap<String, Vec<String>>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            dir: None,
            datasets: Vec::new(),
            epochs: DEFAULT_EPOCHS,
            negative_share: DEFAULT_NEGATIVE_SHARE,
            heldout_share: DEFAULT_HELDOUT_SHARE,
            oversample: DEFAULT_OVERSAMPLE,
            timeout_hours: DEFAULT_TIMEOUT_HOURS,
            synonyms: std::collections::BTreeMap::new(),
        }
    }
}

impl TrainingConfig {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// The training folder for the data root `root`.
    pub fn dir_in(&self, root: &Path) -> PathBuf {
        match &self.dir {
            Some(d) if d.is_absolute() => d.clone(),
            Some(d) => root.join(d),
            None => root.join(DEFAULT_DIR_NAME),
        }
    }

    /// Range-check every field; errors name the key.
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(1..=MAX_EPOCHS).contains(&self.epochs) {
            anyhow::bail!(
                "training.epochs must be between 1 and {MAX_EPOCHS}, got {}",
                self.epochs
            );
        }
        for (key, v) in [
            ("training.negative_share", self.negative_share),
            ("training.heldout_share", self.heldout_share),
        ] {
            if !v.is_finite() || !(0.0..=0.9).contains(&v) {
                anyhow::bail!("{key} must be between 0 and 0.9, got {v}");
            }
        }
        if !(1..=MAX_OVERSAMPLE).contains(&self.oversample) {
            anyhow::bail!(
                "training.oversample must be between 1 and {MAX_OVERSAMPLE}, got {}",
                self.oversample
            );
        }
        if !(1..=MAX_TIMEOUT_HOURS).contains(&self.timeout_hours) {
            anyhow::bail!(
                "training.timeout_hours must be between 1 and {MAX_TIMEOUT_HOURS}, got {}",
                self.timeout_hours
            );
        }
        for d in &self.datasets {
            if d.as_os_str().is_empty() {
                anyhow::bail!("training.datasets has an empty path");
            }
        }
        if self.synonyms.len() > crate::collect::config::MAX_CLASSES {
            anyhow::bail!(
                "training.synonyms has more than {} classes",
                crate::collect::config::MAX_CLASSES
            );
        }
        for (class, phrases) in &self.synonyms {
            let ok = |s: &str| {
                !s.trim().is_empty()
                    && s.chars().count() <= crate::collect::config::MAX_CLASS_LEN
                    && !s.chars().any(|c| c.is_control() || c == '.')
            };
            if !ok(class) || phrases.len() > 16 || !phrases.iter().all(|p| ok(p)) {
                anyhow::bail!(
                    "training.synonyms.{class}: give at most 16 phrases of 1 to {} characters \
                     without periods",
                    crate::collect::config::MAX_CLASS_LEN
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate_and_round_trip() {
        let c = TrainingConfig::default();
        c.validate().unwrap();
        assert!(c.is_default());
        let text = serde_json::to_string(&c).unwrap();
        assert!(text.contains(r#""negative_share":0.15"#), "{text}");
        let back: TrainingConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        // Missing keys take their defaults.
        let partial: TrainingConfig = serde_json::from_str(r#"{"epochs": 3}"#).unwrap();
        assert_eq!(partial.epochs, 3);
        assert_eq!(partial.oversample, DEFAULT_OVERSAMPLE);
    }

    #[test]
    fn validation_names_the_key() {
        let bad = |f: fn(&mut TrainingConfig)| {
            let mut c = TrainingConfig::default();
            f(&mut c);
            c.validate().unwrap_err().to_string()
        };
        assert!(bad(|c| c.epochs = 0).contains("training.epochs"));
        assert!(bad(|c| c.negative_share = 0.95).contains("training.negative_share"));
        assert!(bad(|c| c.heldout_share = f32::NAN).contains("training.heldout_share"));
        assert!(bad(|c| c.oversample = 0).contains("training.oversample"));
        assert!(bad(|c| c.timeout_hours = 0).contains("training.timeout_hours"));
        assert!(bad(|c| c.datasets = vec![PathBuf::new()]).contains("training.datasets"));
        assert!(
            bad(|c| c.synonyms = [("raccoon".to_string(), vec!["a. b".to_string()])].into())
                .contains("training.synonyms")
        );
    }

    /// Every `training.*` key has a row in the README's configuration table.
    #[test]
    fn readme_documents_every_training_key() {
        let readme = include_str!("../../README.md");
        assert!(readme.contains("| `training` |"));
        let c = TrainingConfig {
            dir: Some("x".into()),
            synonyms: [("raccoon".to_string(), vec!["raccoon".to_string()])].into(),
            ..TrainingConfig::default()
        };
        let v = serde_json::to_value(c).unwrap();
        let keys: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
        assert!(keys.contains(&"dir".to_string()));
        let missing: Vec<&String> = keys
            .iter()
            .filter(|k| !readme.contains(&format!("| `training.{k}` |")))
            .collect();
        assert!(
            missing.is_empty(),
            "README.md lacks rows for training.{missing:?}"
        );
    }

    #[test]
    fn dir_resolves_against_the_data_root() {
        let root = Path::new("/data");
        let mut c = TrainingConfig::default();
        assert_eq!(c.dir_in(root), root.join("training"));
        c.dir = Some("runs".into());
        assert_eq!(c.dir_in(root), root.join("runs"));
        c.dir = Some("/abs/runs".into());
        assert_eq!(c.dir_in(root), PathBuf::from("/abs/runs"));
    }
}
