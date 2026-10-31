//! The `training` config section: where runs go, which review exports the Training page offers,
//! and the dataset assembly defaults. Read when a run is queued (every field applies at once);
//! pure data plus validation.

use crate::resources::catalog::TrainFamily;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Folder name of the default training folder under the data root.
pub const DEFAULT_DIR_NAME: &str = "training";
/// The old default `training.epochs`, written out by every save before the per-family
/// recommendations: a stored 50 reads as unset ([`legacy_epochs`]).
pub const LEGACY_DEFAULT_EPOCHS: u32 = 50;
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
/// The Training page's per-epoch table columns when `metric_columns` is empty.
pub const DEFAULT_METRIC_COLUMNS: [&str; 7] =
    ["map", "map50", "f1", "f2", "precision", "recall", "loss"];
/// Most columns of the per-epoch table.
pub const MAX_METRIC_COLUMNS: usize = 20;
/// Most data loader workers `training.workers` may force.
pub const MAX_WORKERS: u32 = 32;

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
    /// Epochs of every run (a cap while early stopping is on); None = the family's
    /// recommendation ([`super::recipe::recommended`]). A stored 50 (the old default every save
    /// wrote) reads as None.
    #[serde(
        deserialize_with = "legacy_epochs",
        skip_serializing_if = "Option::is_none"
    )]
    pub epochs: Option<u32>,
    /// Epochs without improvement before a run stops (0 = off); None = the family's
    /// recommendation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patience: Option<u32>,
    /// Learning rate of every run (above 0, below 1); None = the family's default
    /// ([`super::recipe::default_lr_label`]). A run's form value wins.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr: Option<f32>,
    /// Learning-rate schedule of every run; None = the family's default. A family that does
    /// not take it trains with its default ([`Self::lr_schedule_for`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_schedule: Option<super::recipe::LrSchedule>,
    /// Final learning rate as a share of the starting one (0-1); None = the family's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_min_factor: Option<f32>,
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
    /// Columns of the Training page's per-epoch metrics table, in order: metric ids
    /// (`map`, `f2`, ... see `progress::METRICS`) or a trainer's raw metric names. Empty = the
    /// default ([`DEFAULT_METRIC_COLUMNS`]). Applies at once.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub metric_columns: Vec<String>,
    /// The Training page's and the CLI's default balancing of the training frames
    /// ([`super::dataset::Balance`]); None = off. A run uses what its request says.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub balance: Option<super::dataset::Balance>,
    /// Data loader worker processes of every run; None = automatic (all cores but one, at most 8,
    /// on Windows fewer when the free commit cannot hold them: [`super::workers_for`]). 0 loads
    /// the data in the training process (slower, needs the least memory).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workers: Option<u32>,
    /// Devices the benchmark at the end of a run uses, one after the other (canonical specs:
    /// `ort:cuda:0`, `openvino:gpu.0`, ...). Empty = the one device the model would load on.
    /// Saved from the Train form; a queued run keeps the list it was queued with.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub benchmark_devices: Vec<String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            dir: None,
            datasets: Vec::new(),
            epochs: None,
            patience: None,
            lr: None,
            lr_schedule: None,
            lr_min_factor: None,
            negative_share: DEFAULT_NEGATIVE_SHARE,
            heldout_share: DEFAULT_HELDOUT_SHARE,
            oversample: DEFAULT_OVERSAMPLE,
            timeout_hours: DEFAULT_TIMEOUT_HOURS,
            synonyms: std::collections::BTreeMap::new(),
            metric_columns: Vec::new(),
            balance: None,
            workers: None,
            benchmark_devices: Vec::new(),
        }
    }
}

/// Benchmark device specs in canonical form, deduplicated in order; `auto` and unknown specs are
/// refused (the benchmark runs exact devices).
pub fn benchmark_devices(specs: &[impl AsRef<str>]) -> anyhow::Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for s in specs {
        let s = s.as_ref();
        match crate::backend::spec::parse(s).map_err(|e| anyhow::anyhow!("device '{s}': {e}"))? {
            crate::backend::DeviceSpec::Auto => anyhow::bail!(
                "device 'auto' is not a single device; pick devices, or none for the device the                  model would load on"
            ),
            crate::backend::DeviceSpec::Device(d) => {
                let d = d.to_string();
                if !out.contains(&d) {
                    out.push(d);
                }
            }
        }
    }
    Ok(out)
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

    /// Epochs of a `family` run: `epochs`, else the family's recommendation.
    pub fn epochs_for(&self, family: TrainFamily) -> u32 {
        self.epochs
            .unwrap_or_else(|| super::recipe::recommended(family).epochs)
    }

    /// Patience of a `family` run: `patience`, else the family's recommendation (at most the
    /// run's epochs).
    pub fn patience_for(&self, family: TrainFamily) -> u32 {
        self.patience
            .unwrap_or_else(|| super::recipe::recommended(family).patience)
            .min(self.epochs_for(family))
    }

    /// `lr_schedule` and `lr_min_factor` for a `family` run: each when the family takes it
    /// ([`super::recipe::lr_schedule_error`]), else None (the family's default), so a stored
    /// choice for one family never fails a run of another.
    pub fn lr_schedule_for(
        &self,
        family: TrainFamily,
    ) -> (Option<super::recipe::LrSchedule>, Option<f32>) {
        use super::recipe::lr_schedule_error;
        let schedule = self
            .lr_schedule
            .filter(|&s| lr_schedule_error(family, Some(s), None).is_none());
        let factor = self
            .lr_min_factor
            .filter(|&f| lr_schedule_error(family, schedule, Some(f)).is_none());
        (schedule, factor)
    }

    /// Fill `r`'s learning-rate settings: each typed value (the form's, the CLI's) wins, else
    /// this config's when `r`'s family takes it ([`Self::lr_schedule_for`]; a stored factor
    /// also has to suit the typed schedule), else None (the family's default).
    pub fn apply_lr(
        &self,
        r: &mut super::recipe::TrainRecipe,
        lr: Option<f32>,
        schedule: Option<super::recipe::LrSchedule>,
        min_factor: Option<f32>,
    ) {
        let (cfg_schedule, _) = self.lr_schedule_for(r.family);
        r.lr = lr.or(self.lr);
        r.lr_schedule = schedule.or(cfg_schedule);
        r.lr_min_factor = min_factor.or_else(|| {
            self.lr_min_factor.filter(|&k| {
                super::recipe::lr_schedule_error(r.family, r.lr_schedule, Some(k)).is_none()
            })
        });
    }

    /// The per-epoch table's columns: `metric_columns`, else the default.
    pub fn effective_metric_columns(&self) -> Vec<String> {
        if self.metric_columns.is_empty() {
            DEFAULT_METRIC_COLUMNS
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            self.metric_columns.clone()
        }
    }

    /// Range-check every field; errors name the key.
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(e) = self.epochs
            && !(1..=MAX_EPOCHS).contains(&e)
        {
            anyhow::bail!("training.epochs must be between 1 and {MAX_EPOCHS}, got {e}");
        }
        if let Some(p) = self.patience
            && p > MAX_EPOCHS
        {
            anyhow::bail!("training.patience must be between 0 (off) and {MAX_EPOCHS}, got {p}");
        }
        if let (Some(e), Some(p)) = (self.epochs, self.patience)
            && p > e
        {
            anyhow::bail!("training.patience ({p}) must not exceed training.epochs ({e})");
        }
        if let Some(lr) = self.lr
            && !(lr.is_finite() && lr > 0.0 && lr < 1.0)
        {
            anyhow::bail!("training.lr must be above 0 and below 1 (e.g. 0.0001), got {lr}");
        }
        if let Some(f) = self.lr_min_factor
            && !(f.is_finite() && (0.0..=1.0).contains(&f))
        {
            anyhow::bail!("training.lr_min_factor must be between 0 and 1, got {f}");
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
        validate_metric_columns(&self.metric_columns)?;
        if let Some(w) = self.workers
            && w > MAX_WORKERS
        {
            anyhow::bail!("training.workers must be between 0 and {MAX_WORKERS}, got {w}");
        }
        if let Some(b) = &self.balance {
            b.validate().map_err(|e| anyhow::anyhow!("training.{e}"))?;
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

/// `training.epochs`, with the legacy rule: 50 (the old default every save wrote) = None.
fn legacy_epochs<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u32>, D::Error> {
    Ok(Option::<u32>::deserialize(d)?.filter(|&e| e != LEGACY_DEFAULT_EPOCHS))
}

/// Check a `metric_columns` list: at most [`MAX_METRIC_COLUMNS`] distinct names of 1 to
/// [`super::progress::MAX_METRIC_KEY`] characters without control characters (a known metric id
/// or a trainer's raw metric name).
pub fn validate_metric_columns(cols: &[String]) -> anyhow::Result<()> {
    if cols.len() > MAX_METRIC_COLUMNS {
        anyhow::bail!(
            "training.metric_columns has {} entries; choose at most {MAX_METRIC_COLUMNS}",
            cols.len()
        );
    }
    let max = super::progress::MAX_METRIC_KEY;
    for (i, c) in cols.iter().enumerate() {
        if c.trim().is_empty() || c.trim() != c {
            anyhow::bail!(
                "training.metric_columns has an empty name or one with surrounding spaces"
            );
        }
        if c.chars().count() > max || c.chars().any(char::is_control) {
            anyhow::bail!(
                "training.metric_columns: '{}' is not a metric name (1 to {max} characters, \
                 no control characters)",
                c.escape_debug()
            );
        }
        if cols.iter().take(i).any(|x| x == c) {
            anyhow::bail!("training.metric_columns lists '{c}' twice");
        }
    }
    Ok(())
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
        assert_eq!(partial.epochs, Some(3));
        assert_eq!(partial.oversample, DEFAULT_OVERSAMPLE);
        assert_eq!(partial.workers, None);
        assert!(!text.contains("workers"), "{text}");
        let forced: TrainingConfig = serde_json::from_str(r#"{"workers": 0}"#).unwrap();
        assert_eq!(forced.workers, Some(0));
        forced.validate().unwrap();
        assert_eq!(
            serde_json::from_str::<TrainingConfig>(&serde_json::to_string(&forced).unwrap())
                .unwrap(),
            forced
        );
    }

    #[test]
    fn validation_names_the_key() {
        let bad = |f: fn(&mut TrainingConfig)| {
            let mut c = TrainingConfig::default();
            f(&mut c);
            c.validate().unwrap_err().to_string()
        };
        assert!(bad(|c| c.epochs = Some(0)).contains("training.epochs"));
        assert!(bad(|c| c.patience = Some(MAX_EPOCHS + 1)).contains("training.patience"));
        assert!(
            bad(|c| {
                c.epochs = Some(5);
                c.patience = Some(6);
            })
            .contains("must not exceed")
        );
        assert!(bad(|c| c.negative_share = 0.95).contains("training.negative_share"));
        assert!(bad(|c| c.heldout_share = f32::NAN).contains("training.heldout_share"));
        assert!(bad(|c| c.oversample = 0).contains("training.oversample"));
        assert!(bad(|c| c.timeout_hours = 0).contains("training.timeout_hours"));
        assert!(bad(|c| c.workers = Some(MAX_WORKERS + 1)).contains("training.workers"));
        assert!(bad(|c| c.datasets = vec![PathBuf::new()]).contains("training.datasets"));
        assert!(
            bad(|c| c.synonyms = [("raccoon".to_string(), vec!["a. b".to_string()])].into())
                .contains("training.synonyms")
        );
        assert!(bad(|c| c.metric_columns = vec![String::new()]).contains("metric_columns"));
        assert!(bad(|c| c.metric_columns = vec!["x".repeat(65)]).contains("metric_columns"));
        assert!(bad(|c| c.metric_columns = vec!["a\nb".into()]).contains("metric_columns"));
        assert!(bad(|c| c.metric_columns = vec!["f2".into(), "f2".into()]).contains("twice"));
        let balance = |f: fn(&mut crate::training::dataset::Balance)| {
            let mut c = TrainingConfig::default();
            let mut b = crate::training::dataset::Balance::default();
            f(&mut b);
            c.balance = Some(b);
            c.validate().map_err(|e| e.to_string())
        };
        assert!(balance(|_| {}).is_ok());
        let e = balance(|b| b.strength = 1.5).unwrap_err();
        assert!(e.contains("training.balance.strength"), "{e}");
        assert!(balance(|b| b.strength = f32::NAN).is_err());
        assert!(
            balance(|b| b.max_repeat = 0)
                .unwrap_err()
                .contains("training.balance.max_repeat")
        );
        assert!(balance(|b| b.max_repeat = 11).is_err());
        assert!(
            balance(|b| b.cap_dominant = Some(0.5))
                .unwrap_err()
                .contains("cap_dominant")
        );
        assert!(balance(|b| b.cap_dominant = Some(f32::INFINITY)).is_err());
        assert!(balance(|b| b.cap_dominant = Some(1.0)).is_ok());
        assert!(
            bad(|c| c.metric_columns = (0..21).map(|i| format!("m{i}")).collect())
                .contains("at most 20")
        );
    }

    #[test]
    fn metric_columns_default_round_trip_and_take_raw_names() {
        let mut c = TrainingConfig::default();
        assert_eq!(
            c.effective_metric_columns(),
            ["map", "map50", "f1", "f2", "precision", "recall", "loss"]
        );
        // Not written while empty.
        assert!(
            !serde_json::to_string(&c)
                .unwrap()
                .contains("metric_columns")
        );
        c.metric_columns = vec![
            "f2".into(),
            "val/AP/raccoon".into(),
            "metrics/mAP50(B)".into(),
        ];
        c.validate().unwrap();
        let text = serde_json::to_string(&c).unwrap();
        let back: TrainingConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back.effective_metric_columns(), c.metric_columns);
        // Every known id is a valid column.
        let ids: Vec<String> = crate::training::progress::METRICS
            .iter()
            .take(MAX_METRIC_COLUMNS)
            .map(|m| m.id.to_string())
            .collect();
        validate_metric_columns(&ids).unwrap();
        for d in DEFAULT_METRIC_COLUMNS {
            assert!(crate::training::progress::metric(d).is_some(), "{d}");
        }
    }

    /// Every `training.*` key has a row in the README's configuration table.
    #[test]
    fn readme_documents_every_training_key() {
        let readme = include_str!("../../README.md");
        assert!(readme.contains("| `training` |"));
        let c = TrainingConfig {
            dir: Some("x".into()),
            synonyms: [("raccoon".to_string(), vec!["raccoon".to_string()])].into(),
            metric_columns: vec!["map".into()],
            epochs: Some(30),
            patience: Some(5),
            lr: Some(1e-4),
            lr_schedule: Some(crate::training::recipe::LrSchedule::Cosine),
            lr_min_factor: Some(0.1),
            balance: Some(Default::default()),
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
    fn epochs_and_patience_fall_back_to_the_family_and_legacy_50_is_unset() {
        let c = TrainingConfig::default();
        assert!(!serde_json::to_string(&c).unwrap().contains("epochs"));
        assert_eq!(c.epochs_for(TrainFamily::Yolo26), 100);
        assert_eq!(c.patience_for(TrainFamily::Yolo26), 15);
        assert_eq!(c.epochs_for(TrainFamily::Dfine), 80);
        assert_eq!(c.patience_for(TrainFamily::Rfdetr), 10);
        // The old default, written by every save: the recommendation again.
        let old: TrainingConfig = serde_json::from_str(r#"{"epochs": 50}"#).unwrap();
        assert_eq!(old.epochs, None);
        assert!(old.is_default());
        let null: TrainingConfig = serde_json::from_str(r#"{"epochs": null}"#).unwrap();
        assert_eq!(null.epochs, None);
        let set: TrainingConfig = serde_json::from_str(r#"{"epochs": 30, "patience": 0}"#).unwrap();
        set.validate().unwrap();
        assert_eq!(set.epochs_for(TrainFamily::Rtdetrv2), 30);
        assert_eq!(set.patience_for(TrainFamily::Yolo11), 0);
        let text = serde_json::to_string(&set).unwrap();
        assert!(text.contains(r#""epochs":30,"patience":0"#), "{text}");
        // A recommended patience above a small epochs override is capped by it.
        let small: TrainingConfig = serde_json::from_str(r#"{"epochs": 8}"#).unwrap();
        assert_eq!(small.patience_for(TrainFamily::Yolo26), 8);
    }

    #[test]
    fn learning_rate_settings_validate_round_trip_and_fit_the_family() {
        use crate::training::recipe::LrSchedule;
        let bad = |f: fn(&mut TrainingConfig)| {
            let mut c = TrainingConfig::default();
            f(&mut c);
            c.validate().unwrap_err().to_string()
        };
        for lr in [0.0, 1.0, -1e-4, f32::NAN] {
            let c = TrainingConfig {
                lr: Some(lr),
                ..TrainingConfig::default()
            };
            assert!(
                c.validate()
                    .unwrap_err()
                    .to_string()
                    .contains("training.lr "),
                "{lr}"
            );
        }
        assert!(bad(|c| c.lr_min_factor = Some(1.5)).contains("training.lr_min_factor"));
        assert!(bad(|c| c.lr_min_factor = Some(f32::NAN)).contains("training.lr_min_factor"));
        let c: TrainingConfig =
            serde_json::from_str(r#"{"lr": 0.002, "lr_schedule": "step", "lr_min_factor": 0.2}"#)
                .unwrap();
        c.validate().unwrap();
        assert_eq!(
            (c.lr, c.lr_schedule, c.lr_min_factor),
            (Some(0.002), Some(LrSchedule::Step), Some(0.2))
        );
        let back: TrainingConfig =
            serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
        assert!(serde_json::from_str::<TrainingConfig>(r#"{"lr_schedule": "exp"}"#).is_err());
        // Unset: not written.
        let text = serde_json::to_string(&TrainingConfig::default()).unwrap();
        assert!(!text.contains("\"lr"), "{text}");
        // A schedule the family does not take falls back to its default.
        assert_eq!(
            c.lr_schedule_for(TrainFamily::Rfdetr),
            (Some(LrSchedule::Step), Some(0.2))
        );
        assert_eq!(c.lr_schedule_for(TrainFamily::Yolo26), (None, Some(0.2)));
        let lin = TrainingConfig {
            lr_schedule: Some(LrSchedule::Linear),
            lr_min_factor: Some(0.1),
            ..TrainingConfig::default()
        };
        assert_eq!(
            lin.lr_schedule_for(TrainFamily::Yolo11),
            (Some(LrSchedule::Linear), Some(0.1))
        );
        // transformers' linear schedule ends at 0: the factor is dropped, not the schedule.
        assert_eq!(
            lin.lr_schedule_for(TrainFamily::Dfine),
            (Some(LrSchedule::Linear), None)
        );
        assert_eq!(lin.lr_schedule_for(TrainFamily::Rfdetr), (None, Some(0.1)));
        // Typed values win; the stored ones fill the gaps where they suit.
        let stored = TrainingConfig {
            lr: Some(0.002),
            lr_schedule: Some(LrSchedule::Cosine),
            lr_min_factor: Some(0.05),
            ..TrainingConfig::default()
        };
        let mut r = crate::training::recipe::TrainRecipe::new(TrainFamily::Dfine, "s").unwrap();
        stored.apply_lr(&mut r, None, None, None);
        assert_eq!(
            (r.lr, r.lr_schedule, r.lr_min_factor),
            (Some(0.002), Some(LrSchedule::Cosine), Some(0.05))
        );
        r.validate().unwrap();
        // A typed linear schedule drops the stored factor (D-FINE's linear ends at 0).
        stored.apply_lr(&mut r, Some(1e-4), Some(LrSchedule::Linear), None);
        assert_eq!(
            (r.lr, r.lr_schedule, r.lr_min_factor),
            (Some(1e-4), Some(LrSchedule::Linear), None)
        );
        r.validate().unwrap();
        // Nothing stored, nothing typed: the family's defaults.
        TrainingConfig::default().apply_lr(&mut r, None, None, None);
        assert_eq!((r.lr, r.lr_schedule, r.lr_min_factor), (None, None, None));
    }

    #[test]
    fn balance_defaults_off_and_round_trips() {
        use crate::training::dataset::{Balance, BalanceAxis};
        // An older config has no balance: off, and not written back.
        let old: TrainingConfig = serde_json::from_str(r#"{"oversample": 2}"#).unwrap();
        assert_eq!(old.balance, None);
        assert!(!serde_json::to_string(&old).unwrap().contains("balance"));
        let c: TrainingConfig = serde_json::from_str(
            r#"{"balance": {"axes": ["class", "camera-day"], "strength": 1, "cap_dominant": 3}}"#,
        )
        .unwrap();
        c.validate().unwrap();
        let b = c.balance.clone().unwrap();
        assert_eq!(
            b.axes.iter().copied().collect::<Vec<_>>(),
            [BalanceAxis::Class, BalanceAxis::CameraDay]
        );
        assert_eq!(
            (b.strength, b.max_repeat, b.cap_dominant),
            (1.0, 4, Some(3.0))
        );
        let text = serde_json::to_string(&c).unwrap();
        assert!(
            text.contains(
                r#""axes":["class","camera-day"],"strength":1.0,"max_repeat":4,"cap_dominant":3.0"#
            ),
            "{text}"
        );
        let back: TrainingConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        assert!(Balance::default().is_off());
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
