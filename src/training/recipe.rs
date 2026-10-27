//! [`TrainRecipe`]: what to train (family, size, base weights, input size, epochs, batch,
//! learning rate, night augmentation), its validation, and the trained model's default name.

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
    /// Defaults for `family` / `size`: the size's input size, `epochs`, batch 8 (4 for RF-DETR),
    /// no augmentation. Errors when the size is not one of the family's.
    pub fn new(family: TrainFamily, size: &str, epochs: u32) -> Result<Self> {
        let Some(size) = family.size(size) else {
            bail!(
                "{} has no size '{size}'; choose one of {}",
                family.title(),
                family.sizes().join(", ")
            );
        };
        Ok(Self {
            family,
            size: size.to_string(),
            base_weights: None,
            imgsz: family.default_imgsz(size),
            epochs,
            batch: if family == TrainFamily::Rfdetr { 4 } else { 8 },
            lr: None,
            augment: Augment::default(),
            multi_scale: MultiScale::Auto,
        })
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

    /// Default name of the trained model: `night-yolo26n-20261022-1930` for a night / IR run,
    /// `custom-...` otherwise (`stamp` = `YYYYMMDD-HHMM`).
    pub fn default_name(&self, night: bool, stamp: &str) -> String {
        format!(
            "{}-{}-{stamp}",
            if night { "night" } else { "custom" },
            self.label()
        )
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
        let r = TrainRecipe::new(TrainFamily::Rfdetr, "small", 10).unwrap();
        assert_eq!((r.size.as_str(), r.imgsz, r.batch), ("small", 512, 4));
        assert_eq!(r.label(), "rfdetr-small");
        assert_eq!(r.base().unwrap().id, "train-base:rfdetr-small");
        r.validate().unwrap();
        let y = TrainRecipe::new(TrainFamily::Yolo26, "nano", 1).unwrap();
        assert_eq!((y.size.as_str(), y.imgsz, y.batch), ("n", 640, 8));
        assert_eq!(y.base().unwrap().id, "model:yolo26n");
        assert!(TrainRecipe::new(TrainFamily::Dfine, "n", 1).is_err());
        assert_eq!(
            y.default_name(true, "20261022-1930"),
            "night-yolo26n-20261022-1930"
        );
        assert_eq!(
            TrainRecipe::new(TrainFamily::Rtdetrv2, "s", 1)
                .unwrap()
                .default_name(false, "x"),
            "custom-rt-detrv2-s-x"
        );
    }

    #[test]
    fn validation_rejects_bad_values() {
        let base = TrainRecipe::new(TrainFamily::Yolo11, "s", 5).unwrap();
        let bad = |f: fn(&mut TrainRecipe)| {
            let mut r = base.clone();
            f(&mut r);
            r.validate().unwrap_err().to_string()
        };
        assert!(bad(|r| r.imgsz = 650).contains("multiple of 32"));
        assert!(bad(|r| r.imgsz = 64).contains("between 160"));
        assert!(bad(|r| r.epochs = 0).contains("epochs"));
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
        let mut rf = TrainRecipe::new(TrainFamily::Rfdetr, "base", 5).unwrap();
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
        let mut r = TrainRecipe::new(TrainFamily::Rtdetrv2, "m", 30).unwrap();
        r.lr = Some(2e-4);
        r.augment = Augment::NIGHT;
        let text = serde_json::to_string(&r).unwrap();
        assert!(text.contains(r#""family":"rtdetrv2""#), "{text}");
        let back: TrainRecipe = serde_json::from_str(&text).unwrap();
        assert_eq!(back, r);
        assert!(!r.augment.is_off());
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
