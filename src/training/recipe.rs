//! [`TrainRecipe`]: what to train (family, size, base weights, input size, epochs, patience,
//! batch, learning rate, night augmentation), its validation, the per-family starting points
//! ([`recommended`]) and the trained model's default name.

use crate::resources::catalog::{self, Resource, TrainFamily};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Most images per batch.
pub const MAX_BATCH: u32 = 256;
/// Input size bounds.
pub const MIN_IMGSZ: u32 = 160;
pub const MAX_IMGSZ: u32 = 1536;

/// Synthetic night augmentation, as per-image probabilities (applied by the trainer, see
/// `scripts/train_model.py`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Augment {
    /// Under-exposure (exposure x0.05-0.4 in linear light).
    pub dark: f32,
    /// Shot and read noise.
    pub noise: f32,
    /// Grayscale (IR-like).
    pub gray: f32,
}

impl Augment {
    /// The night preset (plan: darkness, noise and grayscale at p = 0.5, 0.3, 0.5).
    pub const NIGHT: Augment = Augment {
        dark: 0.5,
        noise: 0.3,
        gray: 0.5,
    };

    pub fn is_off(&self) -> bool {
        self.dark == 0.0 && self.noise == 0.0 && self.gray == 0.0
    }
}

/// RF-DETR multi-scale training (rfdetr resizes every batch to one of ~11 sizes up to about
/// 1.4x the input size, see [`super::estimate::rfdetr_max_side`]): better accuracy, about twice
/// the memory of the input size alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MultiScale {
    /// On when the estimate fits a batch of at least [`MULTI_SCALE_MIN_BATCH`] with it on the
    /// run's GPU, else off ([`TrainRecipe::multi_scale_on`]).
    #[default]
    Auto,
    On,
    Off,
}

/// Smallest batch `auto` keeps multi-scale training for.
pub const MULTI_SCALE_MIN_BATCH: u32 = 4;

impl MultiScale {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "on" | "true" | "1" => Some(Self::On),
            "off" | "false" | "0" => Some(Self::Off),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }

    pub fn is_auto(&self) -> bool {
        *self == Self::Auto
    }
}

/// A family's starting point for fine-tuning: with early stopping on, `epochs` is a cap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recommended {
    pub epochs: u32,
    /// Epochs without an improvement of more than `min_delta` before the run stops.
    pub patience: u32,
    /// No convergence stop before this epoch.
    pub min_epochs: u32,
    /// Smallest validation mAP gain that counts as an improvement.
    pub min_delta: f32,
    /// The gain is measured on the mean of the last `smooth` validation mAP values, so one noisy
    /// peak of a small validation split does not start the countdown (1 = the raw value).
    pub smooth: u32,
    /// Learning rate; None = the trainer's default (rfdetr and the transformers recipe: 1e-4).
    pub lr: Option<f32>,
}

impl Recommended {
    /// `min_delta` as the f64 the training script parses from its decimal text (`0.001`), so the
    /// display twin of its rule ([`super::progress::convergence_decision`]) decides alike.
    pub fn min_delta_f64(&self) -> f64 {
        self.min_delta.to_string().parse().unwrap_or(0.0)
    }
}

/// The starting point of `family`. Sources: the Ultralytics fine-tuning guide
/// (<https://docs.ultralytics.com/guides/finetuning-guide>: patience 10-20 when fine-tuning) and
/// the RF-DETR training docs (<https://rfdetr.roboflow.com/learn/train/>: 100 epochs, lr 1e-4,
/// patience 10, min_delta 0.001); RT-DETRv2 / D-FINE converge in fewer epochs from COCO weights.
pub const fn recommended(family: TrainFamily) -> Recommended {
    let (epochs, patience) = match family {
        TrainFamily::Yolo26 | TrainFamily::Yolo11 | TrainFamily::Yolov8 => (100, 15),
        TrainFamily::Rfdetr => (100, 10),
        TrainFamily::Rtdetrv2 | TrainFamily::Dfine => (80, 10),
    };
    Recommended {
        epochs,
        patience,
        min_epochs: 10,
        min_delta: 0.001,
        smooth: 3,
        lr: None,
    }
}

/// Validation splits below this many images make the per-epoch mAP noisy (run
/// 20261030-010233-b69d: 16 images, +-0.05 between epochs): a recommended patience doubles there.
pub const SMALL_VALID_IMAGES: usize = 50;

/// One training run's model choice and hyper-parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainRecipe {
    pub family: TrainFamily,
    /// As the family spells it (`n`, `nano`, `s`, ...).
    pub size: String,
    /// Base weights resource id (`model:yolo26n`, `train-base:rfdetr-nano`); None = the family's
    /// pin for the size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_weights: Option<String>,
    pub imgsz: u32,
    pub epochs: u32,
    /// Stop after this many epochs without a validation mAP improvement (0 = off; a recipe
    /// queued before early stopping existed reads as 0, its old behaviour).
    #[serde(default)]
    pub patience: u32,
    /// `patience` is the family's recommendation (nothing typed, `training.patience` unset), so a
    /// small validation split may double it ([`Self::convergence_patience`]); a typed patience is
    /// never changed (a recipe queued before this field existed reads as typed).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub patience_recommended: bool,
    pub batch: u32,
    /// Learning rate; None = the trainer's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr: Option<f32>,
    #[serde(default)]
    pub augment: Augment,
    /// RF-DETR only (other families ignore it).
    #[serde(default, skip_serializing_if = "MultiScale::is_auto")]
    pub multi_scale: MultiScale,
}

impl TrainRecipe {
    /// Defaults for `family` / `size`: the size's input size, the [`recommended`] epochs and
    /// patience, batch 8 (4 for RF-DETR), no augmentation. Errors when the size is not one of the
    /// family's.
    pub fn new(family: TrainFamily, size: &str) -> Result<Self> {
        let Some(size) = family.size(size) else {
            bail!(
                "{} has no size '{size}'; choose one of {}",
                family.title(),
                family.sizes().join(", ")
            );
        };
        let rec = recommended(family);
        Ok(Self {
            family,
            size: size.to_string(),
            base_weights: None,
            imgsz: family.default_imgsz(size),
            epochs: rec.epochs,
            patience: rec.patience,
            patience_recommended: true,
            batch: if family == TrainFamily::Rfdetr { 4 } else { 8 },
            lr: rec.lr,
            augment: Augment::default(),
            multi_scale: MultiScale::Auto,
        })
    }

    /// The patience the run trains with on `valid_images` validation images, and the `train.log`
    /// line that explains the rule: a recommended patience doubles below [`SMALL_VALID_IMAGES`]
    /// (at most the epochs); a typed one is kept.
    pub fn convergence_patience(&self, valid_images: usize) -> (u32, String) {
        let smooth = recommended(self.family).smooth;
        let window = if smooth > 1 {
            format!("mAP averaged over the last {smooth} epochs")
        } else {
            "mAP of each epoch".into()
        };
        if self.patience_recommended && self.patience > 0 && valid_images < SMALL_VALID_IMAGES {
            let p = self.patience.saturating_mul(2).min(self.epochs);
            if p != self.patience {
                return (
                    p,
                    format!(
                        "convergence: {valid_images} validation images (< {SMALL_VALID_IMAGES}): \
                         patience {p} instead of {}; {window}",
                        self.patience
                    ),
                );
            }
        }
        let typed = if self.patience_recommended {
            ""
        } else {
            " (set)"
        };
        (
            self.patience,
            format!(
                "convergence: {valid_images} validation images: patience {}{typed}; {window}",
                self.patience
            ),
        )
    }

    /// Whether the convergence stop can end the run before its last epoch (patience on and
    /// below the epochs); else the run gets a warning that early stopping is off.
    pub fn early_stopping(&self) -> bool {
        self.patience > 0 && self.patience < self.epochs
    }

    /// Whether RF-DETR trains multi-scale on a GPU with `memory_mb` for the estimate (0 =
    /// unknown): `on` / `off` as set; `auto` = on when the estimate with it on fits a batch of at
    /// least [`MULTI_SCALE_MIN_BATCH`]. Always false for the other families.
    pub fn multi_scale_on(&self, memory_mb: u64) -> bool {
        if self.family != TrainFamily::Rfdetr {
            return false;
        }
        match self.multi_scale {
            MultiScale::On => true,
            MultiScale::Off => false,
            MultiScale::Auto => {
                memory_mb == 0
                    || super::estimate::suggest_batch_with(
                        self.family,
                        &self.size,
                        self.imgsz,
                        memory_mb,
                        true,
                    )
                    .is_some_and(|b| b >= MULTI_SCALE_MIN_BATCH)
            }
        }
    }

    /// `yolo26n`, `rfdetr-nano`, `rt-detrv2-s`, `dfine-s`.
    pub fn label(&self) -> String {
        self.family.base_name(&self.size)
    }

    /// The base weights resource (explicit, else the family's pin).
    pub fn base(&self) -> Result<&'static Resource> {
        match &self.base_weights {
            Some(id) => {
                let Some(r) = catalog::find_base_weights(id) else {
                    bail!("unknown base weights '{id}'");
                };
                let ok = match self.family.trainer() {
                    catalog::Trainer::Ultralytics => r.model_name().is_some_and(|n| {
                        n.starts_with(match self.family {
                            TrainFamily::Yolo26 => "yolo26",
                            TrainFamily::Yolo11 => "yolo11",
                            _ => "yolov8",
                        })
                    }),
                    _ => r.id.contains(match self.family {
                        TrainFamily::Rfdetr => ":rfdetr-",
                        TrainFamily::Rtdetrv2 => ":rtdetrv2-",
                        _ => ":dfine-",
                    }),
                };
                if !ok {
                    bail!(
                        "base weights '{id}' are not {} weights",
                        self.family.title()
                    );
                }
                Ok(r)
            }
            None => catalog::base_weights(self.family, &self.size).ok_or_else(|| {
                anyhow::anyhow!("no base weights for {} {}", self.family.title(), self.size)
            }),
        }
    }

    /// Check every field; errors say what is allowed.
    pub fn validate(&self) -> Result<()> {
        let Some(size) = self.family.size(&self.size) else {
            bail!(
                "{} has no size '{}'; choose one of {}",
                self.family.title(),
                self.size,
                self.family.sizes().join(", ")
            );
        };
        if size != self.size {
            bail!("size '{}' must be spelled '{size}'", self.size);
        }
        let step = self.family.imgsz_step(size);
        if !(MIN_IMGSZ..=MAX_IMGSZ).contains(&self.imgsz) || !self.imgsz.is_multiple_of(step) {
            bail!(
                "input size {} must be a multiple of {step} between {MIN_IMGSZ} and {MAX_IMGSZ} \
                 (default {})",
                self.imgsz,
                self.family.default_imgsz(size)
            );
        }
        if !(1..=super::config::MAX_EPOCHS).contains(&self.epochs) {
            bail!(
                "epochs must be between 1 and {}, got {}",
                super::config::MAX_EPOCHS,
                self.epochs
            );
        }
        if self.patience > self.epochs {
            bail!(
                "patience ({}) must not exceed the epochs ({}); 0 turns early stopping off",
                self.patience,
                self.epochs
            );
        }
        if !(1..=MAX_BATCH).contains(&self.batch) {
            bail!(
                "batch must be between 1 and {MAX_BATCH}, got {}",
                self.batch
            );
        }
        if let Some(lr) = self.lr
            && !(lr.is_finite() && lr > 0.0 && lr < 1.0)
        {
            bail!("learning rate must be above 0 and below 1, got {lr}");
        }
        for (k, p) in [
            ("darkness", self.augment.dark),
            ("noise", self.augment.noise),
            ("grayscale", self.augment.gray),
        ] {
            if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
                bail!("{k} augmentation probability must be between 0 and 1, got {p}");
            }
        }
        self.base()?;
        Ok(())
    }

    /// Default name of the trained model: `<prefix>-yolo26n-20261022-1930`, the prefix being the
    /// scene filter's label (`custom`, `day`, `night`, `ir`, ... see `SceneFilter::label`;
    /// `stamp` = `YYYYMMDD-HHMM`).
    pub fn default_name(&self, prefix: &str, stamp: &str) -> String {
        format!("{prefix}-{}-{stamp}", self.label())
    }
}

/// `(family, size)` of a model label: `yolo26n`, `yolo11s`, `yolov8m`, `rfdetr-nano`,
/// `rf-detr-small`, `rt-detrv2-s`, `rtdetrv2-m`, `dfine-l`, `d-fine-x` (case-insensitive).
pub fn parse_label(label: &str) -> Option<(TrainFamily, &'static str)> {
    let l = label.trim().to_ascii_lowercase().replace('_', "-");
    for (prefix, family) in [
        ("yolo26", TrainFamily::Yolo26),
        ("yolo11", TrainFamily::Yolo11),
        ("yolov8", TrainFamily::Yolov8),
        ("rf-detr-", TrainFamily::Rfdetr),
        ("rfdetr-", TrainFamily::Rfdetr),
        ("rt-detrv2-", TrainFamily::Rtdetrv2),
        ("rtdetrv2-", TrainFamily::Rtdetrv2),
        ("d-fine-", TrainFamily::Dfine),
        ("dfine-", TrainFamily::Dfine),
    ] {
        if let Some(rest) = l.strip_prefix(prefix) {
            let rest = rest.trim_start_matches('-');
            return family.size(rest).map(|s| (family, s));
        }
    }
    None
}

/// A model name that is safe in a URL path and as a file stem.
pub fn valid_model_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && !name.starts_with('.');
    if !ok {
        bail!(
            "model name '{name}' must be 1-64 characters of letters, digits, '-', '_' and '.' (not \
             starting with '.')"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_the_family() {
        let r = TrainRecipe::new(TrainFamily::Rfdetr, "small").unwrap();
        assert_eq!((r.size.as_str(), r.imgsz, r.batch), ("small", 512, 4));
        assert_eq!(r.label(), "rfdetr-small");
        assert_eq!(r.base().unwrap().id, "train-base:rfdetr-small");
        r.validate().unwrap();
        let y = TrainRecipe::new(TrainFamily::Yolo26, "nano").unwrap();
        assert_eq!((y.size.as_str(), y.imgsz, y.batch), ("n", 640, 8));
        assert_eq!(y.base().unwrap().id, "model:yolo26n");
        assert!(TrainRecipe::new(TrainFamily::Dfine, "n").is_err());
        assert_eq!(
            y.default_name("night", "20261022-1930"),
            "night-yolo26n-20261022-1930"
        );
        assert_eq!(
            TrainRecipe::new(TrainFamily::Rtdetrv2, "s")
                .unwrap()
                .default_name("custom", "x"),
            "custom-rt-detrv2-s-x"
        );
    }

    #[test]
    fn validation_rejects_bad_values() {
        let base = TrainRecipe::new(TrainFamily::Yolo11, "s").unwrap();
        let bad = |f: fn(&mut TrainRecipe)| {
            let mut r = base.clone();
            f(&mut r);
            r.validate().unwrap_err().to_string()
        };
        assert!(bad(|r| r.imgsz = 650).contains("multiple of 32"));
        assert!(bad(|r| r.imgsz = 64).contains("between 160"));
        assert!(bad(|r| r.epochs = 0).contains("epochs"));
        assert!(bad(|r| r.patience = r.epochs + 1).contains("patience"));
        assert!(bad(|r| r.batch = 0).contains("batch"));
        assert!(bad(|r| r.lr = Some(2.0)).contains("learning rate"));
        assert!(bad(|r| r.lr = Some(f32::NAN)).contains("learning rate"));
        assert!(bad(|r| r.augment.gray = 1.5).contains("grayscale"));
        assert!(bad(|r| r.size = "small".into()).contains("spelled 's'"));
        assert!(bad(|r| r.size = "q".into()).contains("no size"));
        assert!(
            bad(|r| r.base_weights = Some("train-base:rfdetr-nano".into())).contains("not YOLO11")
        );
        assert!(bad(|r| r.base_weights = Some("model:yolo26s".into())).contains("not YOLO11"));
        assert!(bad(|r| r.base_weights = Some("nope".into())).contains("unknown base weights"));
        let mut ok = base.clone();
        ok.base_weights = Some("model:yolo11m".into());
        ok.validate().unwrap();
        // RF-DETR base: patch 14 x 4 windows.
        let mut rf = TrainRecipe::new(TrainFamily::Rfdetr, "base").unwrap();
        rf.imgsz = 640;
        assert!(
            rf.validate()
                .unwrap_err()
                .to_string()
                .contains("multiple of 56")
        );
        rf.imgsz = 672;
        rf.validate().unwrap();
    }

    #[test]
    fn recipe_round_trips_as_json() {
        let mut r = TrainRecipe::new(TrainFamily::Rtdetrv2, "m").unwrap();
        r.lr = Some(2e-4);
        r.augment = Augment::NIGHT;
        let text = serde_json::to_string(&r).unwrap();
        assert!(text.contains(r#""family":"rtdetrv2""#), "{text}");
        let back: TrainRecipe = serde_json::from_str(&text).unwrap();
        assert_eq!(back, r);
        assert!(!r.augment.is_off());
    }

    #[test]
    fn recommended_starting_points_per_family() {
        let row = |f| {
            let r = recommended(f);
            (r.epochs, r.patience, r.min_epochs, r.min_delta, r.lr)
        };
        for f in [
            TrainFamily::Yolo26,
            TrainFamily::Yolo11,
            TrainFamily::Yolov8,
        ] {
            assert_eq!(row(f), (100, 15, 10, 0.001, None), "{f}");
        }
        assert_eq!(row(TrainFamily::Rfdetr), (100, 10, 10, 0.001, None));
        for f in [TrainFamily::Rtdetrv2, TrainFamily::Dfine] {
            assert_eq!(row(f), (80, 10, 10, 0.001, None), "{f}");
        }
        assert_eq!(recommended(TrainFamily::Rfdetr).min_delta_f64(), 0.001);
        for f in TrainFamily::ALL {
            let r = TrainRecipe::new(f, f.sizes()[0]).unwrap();
            let rec = recommended(f);
            assert_eq!(
                (r.epochs, r.patience, r.lr),
                (rec.epochs, rec.patience, rec.lr)
            );
            assert!(r.early_stopping(), "{f}");
            r.validate().unwrap();
            assert_eq!(rec.smooth, 3, "{f}");
        }
    }

    #[test]
    fn a_small_validation_split_doubles_only_a_recommended_patience() {
        let r = TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap();
        assert!(r.patience_recommended);
        assert_eq!(
            r.convergence_patience(16),
            (
                20,
                "convergence: 16 validation images (< 50): patience 20 instead of 10; mAP \
                 averaged over the last 3 epochs"
                    .into()
            )
        );
        // 50 images and more: as recommended.
        assert_eq!(
            r.convergence_patience(50),
            (
                10,
                "convergence: 50 validation images: patience 10; mAP averaged over the last 3 \
                 epochs"
                    .into()
            )
        );
        // Capped by the epochs.
        let short = TrainRecipe {
            epochs: 15,
            ..r.clone()
        };
        assert_eq!(short.convergence_patience(16).0, 15);
        // A typed patience (also one equal to the recommendation) is kept; so is 0 (off).
        let typed = TrainRecipe {
            patience_recommended: false,
            ..r.clone()
        };
        let (p, line) = typed.convergence_patience(16);
        assert_eq!(p, 10);
        assert!(line.contains("patience 10 (set)"), "{line}");
        let off = TrainRecipe {
            patience: 0,
            ..r.clone()
        };
        assert_eq!(off.convergence_patience(16).0, 0);
        // The flag is written only when set; an old recipe reads as typed.
        let mut v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["patience_recommended"], true);
        v.as_object_mut().unwrap().remove("patience_recommended");
        let old: TrainRecipe = serde_json::from_value(v).unwrap();
        assert!(!old.patience_recommended);
        assert!(
            !serde_json::to_string(&typed)
                .unwrap()
                .contains("patience_recommended")
        );
    }

    #[test]
    fn patience_bounds_and_early_stopping() {
        let mut r = TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap();
        r.epochs = 10;
        r.patience = 10;
        r.validate().unwrap();
        assert!(!r.early_stopping(), "patience = epochs never fires");
        r.patience = 0;
        r.validate().unwrap();
        assert!(!r.early_stopping());
        r.patience = 3;
        assert!(r.early_stopping());
        r.patience = 11;
        assert!(
            r.validate()
                .unwrap_err()
                .to_string()
                .contains("patience (11)")
        );
        // A recipe queued before `patience` existed: off.
        let mut v = serde_json::to_value(&r).unwrap();
        v.as_object_mut().unwrap().remove("patience");
        let old: TrainRecipe = serde_json::from_value(v).unwrap();
        assert_eq!(old.patience, 0);
    }

    #[test]
    fn labels_parse_to_family_and_size() {
        assert_eq!(parse_label("yolo26n"), Some((TrainFamily::Yolo26, "n")));
        assert_eq!(parse_label("YOLOv8m"), Some((TrainFamily::Yolov8, "m")));
        assert_eq!(parse_label("yolo11-x"), Some((TrainFamily::Yolo11, "x")));
        assert_eq!(
            parse_label("rfdetr-nano"),
            Some((TrainFamily::Rfdetr, "nano"))
        );
        assert_eq!(
            parse_label("rf-detr-s"),
            Some((TrainFamily::Rfdetr, "small"))
        );
        assert_eq!(
            parse_label("rt-detrv2-s"),
            Some((TrainFamily::Rtdetrv2, "s"))
        );
        assert_eq!(parse_label("d-fine-l"), Some((TrainFamily::Dfine, "l")));
        assert_eq!(parse_label("deimv2-s"), None);
        assert_eq!(parse_label("rfdetr-xl"), None);
        assert_eq!(parse_label("yolo26q"), None);
        for f in TrainFamily::ALL {
            for s in f.sizes() {
                assert_eq!(parse_label(&f.base_name(s)), Some((f, *s)), "{f} {s}");
            }
        }
    }

    #[test]
    fn model_names_are_url_safe() {
        valid_model_name("night-yolo26n-20261022-1930").unwrap();
        for bad in ["", ".hidden", "a/b", "a b", "x?y", &"a".repeat(65)] {
            assert!(valid_model_name(bad).is_err(), "{bad}");
        }
    }
}
