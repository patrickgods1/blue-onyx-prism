//! The `collect` config section (dataset collector, [`super`]): which buckets are kept, how
//! often per camera, the dedup distance and the disk caps. Every field applies live. Pure data
//! plus validation; [`CollectSettings`] is the resolved form the workers carry in their live
//! settings.

use super::Bucket;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::{Path, PathBuf};

/// Default total size cap of the dataset folder, in GB.
pub const DEFAULT_MAX_GB: f32 = 5.0;
/// Largest accepted `max_gb`.
pub const MAX_MAX_GB: f32 = 10_000.0;
/// Default age cap, in days.
pub const DEFAULT_MAX_AGE_DAYS: u32 = 90;
/// Largest accepted `max_age_days` (0 = keep forever).
pub const MAX_MAX_AGE_DAYS: u32 = 3650;
/// Default dHash Hamming distance at or below which a frame counts as a near duplicate.
pub const DEFAULT_DEDUP_HAMMING: u32 = 6;
/// Largest accepted `dedup_hamming` (of 64 bits; more would drop unrelated frames).
pub const MAX_DEDUP_HAMMING: u32 = 32;
/// Default `near_threshold.margin`.
pub const DEFAULT_NEAR_MARGIN: f32 = 0.1;
/// Largest accepted `near_threshold.margin`.
pub const MAX_NEAR_MARGIN: f32 = 0.5;
/// Largest accepted `per_camera_min_interval_s` (one week).
pub const MAX_INTERVAL_S: u32 = 7 * 24 * 3600;
/// Folder name of the default dataset directory under the data root.
pub const DEFAULT_DIR_NAME: &str = "dataset";
/// Default review / training class list (`collect.classes`).
pub const DEFAULT_CLASSES: [&str; 14] = [
    "person",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "dog",
    "cat",
    "raccoon",
    "opossum",
    "skunk",
    "fox",
    "deer",
    "squirrel",
    "bird",
];
/// Most classes in `collect.classes`.
pub const MAX_CLASSES: usize = 200;
/// Longest class name.
pub const MAX_CLASS_LEN: usize = 64;

/// [`DEFAULT_CLASSES`] as owned strings.
pub fn default_classes() -> Vec<String> {
    DEFAULT_CLASSES.iter().map(|c| (*c).to_string()).collect()
}

/// One bucket's settings.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BucketConfig {
    pub enabled: bool,
    /// Share of the matching frames kept (0..=1, sampled by request id).
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub rate: f32,
    /// At most one frame of this bucket per camera in this many seconds (0 = no limit).
    pub per_camera_min_interval_s: u32,
    /// `near_threshold` only: the band around the threshold (see [`super::decide`]).
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::config::thr::ser_opt_clean"
    )]
    pub margin: Option<f32>,
}

impl BucketConfig {
    /// The default settings of `bucket`: every bucket on and fully sampled; `empty` at most one
    /// frame per camera per 10 min, `positive` one per 30 s; `near_threshold` with a 0.1 band.
    pub fn default_for(bucket: Bucket) -> Self {
        Self {
            enabled: true,
            rate: 1.0,
            per_camera_min_interval_s: match bucket {
                Bucket::Empty => 600,
                Bucket::Positive => 30,
                _ => 0,
            },
            margin: (bucket == Bucket::NearThreshold).then_some(DEFAULT_NEAR_MARGIN),
        }
    }
}

/// The settings of every bucket.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Buckets {
    pub positive: BucketConfig,
    pub empty: BucketConfig,
    pub suppressed: BucketConfig,
    pub verify_rejected: BucketConfig,
    pub verify_kept: BucketConfig,
    pub near_threshold: BucketConfig,
    pub disagreement: BucketConfig,
}

impl Default for Buckets {
    fn default() -> Self {
        Self {
            positive: BucketConfig::default_for(Bucket::Positive),
            empty: BucketConfig::default_for(Bucket::Empty),
            suppressed: BucketConfig::default_for(Bucket::Suppressed),
            verify_rejected: BucketConfig::default_for(Bucket::VerifyRejected),
            verify_kept: BucketConfig::default_for(Bucket::VerifyKept),
            near_threshold: BucketConfig::default_for(Bucket::NearThreshold),
            disagreement: BucketConfig::default_for(Bucket::Disagreement),
        }
    }
}

impl Buckets {
    pub fn get(&self, b: Bucket) -> &BucketConfig {
        match b {
            Bucket::Positive => &self.positive,
            Bucket::Empty => &self.empty,
            Bucket::Suppressed => &self.suppressed,
            Bucket::VerifyRejected => &self.verify_rejected,
            Bucket::VerifyKept => &self.verify_kept,
            Bucket::NearThreshold => &self.near_threshold,
            Bucket::Disagreement => &self.disagreement,
        }
    }

    pub fn get_mut(&mut self, b: Bucket) -> &mut BucketConfig {
        match b {
            Bucket::Positive => &mut self.positive,
            Bucket::Empty => &mut self.empty,
            Bucket::Suppressed => &mut self.suppressed,
            Bucket::VerifyRejected => &mut self.verify_rejected,
            Bucket::VerifyKept => &mut self.verify_kept,
            Bucket::NearThreshold => &mut self.near_threshold,
            Bucket::Disagreement => &mut self.disagreement,
        }
    }

    /// The `near_threshold` band.
    pub fn near_margin(&self) -> f32 {
        self.near_threshold.margin.unwrap_or(DEFAULT_NEAR_MARGIN)
    }
}

/// A bucket as written in the config: every field optional, missing ones take the bucket's
/// own default ([`BucketConfig::default_for`]), so `{"empty": {"enabled": false}}` keeps the
/// 600 s interval.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawBucket {
    enabled: Option<bool>,
    rate: Option<f32>,
    per_camera_min_interval_s: Option<u32>,
    margin: Option<f32>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawBuckets {
    positive: Option<RawBucket>,
    empty: Option<RawBucket>,
    suppressed: Option<RawBucket>,
    verify_rejected: Option<RawBucket>,
    verify_kept: Option<RawBucket>,
    near_threshold: Option<RawBucket>,
    disagreement: Option<RawBucket>,
}

fn fill(bucket: Bucket, raw: Option<RawBucket>) -> BucketConfig {
    let d = BucketConfig::default_for(bucket);
    let r = raw.unwrap_or_default();
    BucketConfig {
        enabled: r.enabled.unwrap_or(d.enabled),
        rate: r.rate.unwrap_or(d.rate),
        per_camera_min_interval_s: r
            .per_camera_min_interval_s
            .unwrap_or(d.per_camera_min_interval_s),
        // Kept as written (validation refuses a margin on another bucket).
        margin: r.margin.or(d.margin),
    }
}

impl<'de> Deserialize<'de> for Buckets {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let r = RawBuckets::deserialize(d)?;
        Ok(Self {
            positive: fill(Bucket::Positive, r.positive),
            empty: fill(Bucket::Empty, r.empty),
            suppressed: fill(Bucket::Suppressed, r.suppressed),
            verify_rejected: fill(Bucket::VerifyRejected, r.verify_rejected),
            verify_kept: fill(Bucket::VerifyKept, r.verify_kept),
            near_threshold: fill(Bucket::NearThreshold, r.near_threshold),
            disagreement: fill(Bucket::Disagreement, r.disagreement),
        })
    }
}

/// `collect` in the config (see the README and `docs/DATASET.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CollectConfig {
    /// Save frames at all (default off).
    pub enabled: bool,
    /// Dataset folder; None = `<data root>/dataset`. Relative paths resolve against the data
    /// root.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
    /// Folder the Review page, its exports and the auto-labeler work on (an import saved
    /// elsewhere); None = the dataset folder. Relative paths resolve against the data root.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_dir: Option<PathBuf>,
    pub buckets: Buckets,
    /// Total size cap of the folder in GB; the oldest frames are deleted first.
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub max_gb: f32,
    /// Frames older than this many days are deleted (0 = keep forever).
    pub max_age_days: u32,
    /// dHash Hamming distance (of 64 bits) at or below which a frame of the same camera within
    /// the dedup window counts as a near duplicate and is dropped.
    pub dedup_hamming: u32,
    /// Classes of the Review page and the label exports (YOLO class ids follow this order).
    pub classes: Vec<String>,
}

impl Default for CollectConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dir: None,
            review_dir: None,
            buckets: Buckets::default(),
            max_gb: DEFAULT_MAX_GB,
            max_age_days: DEFAULT_MAX_AGE_DAYS,
            dedup_hamming: DEFAULT_DEDUP_HAMMING,
            classes: default_classes(),
        }
    }
}

impl CollectConfig {
    /// Serialized only when it differs from the default (old configs stay unchanged).
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// The dataset folder: `dir` resolved against `data_root`, or `<data_root>/dataset`.
    pub fn resolved_dir(&self, data_root: &Path) -> PathBuf {
        match self.dir.as_deref().filter(|d| !d.as_os_str().is_empty()) {
            Some(d) if d.is_absolute() => d.to_path_buf(),
            Some(d) => data_root.join(d),
            None => data_root.join(DEFAULT_DIR_NAME),
        }
    }

    /// The Review page's folder: `review_dir` resolved against `data_root`, or the dataset
    /// folder ([`Self::resolved_dir`]).
    pub fn resolved_review_dir(&self, data_root: &Path) -> PathBuf {
        match self
            .review_dir
            .as_deref()
            .filter(|d| !d.as_os_str().is_empty())
        {
            Some(d) if d.is_absolute() => d.to_path_buf(),
            Some(d) => data_root.join(d),
            None => self.resolved_dir(data_root),
        }
    }

    /// The size cap in bytes.
    pub fn max_bytes(&self) -> u64 {
        let b = f64::from(self.max_gb.max(0.0)) * 1024.0 * 1024.0 * 1024.0;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "validated to 0..=10000 GB, far below u64::MAX; non-negative by max(0)"
        )]
        let out = b.min(u64::MAX as f64) as u64;
        out
    }

    /// Ranges of every field; `data_root` resolves and checks `dir` like the other folder
    /// settings ([`crate::config::validate_dir_setting`]).
    pub fn validate(&self, data_root: &Path) -> anyhow::Result<()> {
        use anyhow::bail;
        if let Some(d) = &self.dir {
            crate::config::validate_dir_setting("collect.dir", &d.to_string_lossy(), data_root)?;
        }
        if let Some(d) = &self.review_dir {
            crate::config::validate_dir_setting(
                "collect.review_dir",
                &d.to_string_lossy(),
                data_root,
            )?;
        }
        if !(self.max_gb.is_finite() && self.max_gb > 0.0 && self.max_gb <= MAX_MAX_GB) {
            bail!(
                "collect.max_gb: must be above 0 and at most {MAX_MAX_GB}, got {}; the dataset \
                 needs a size cap (default {DEFAULT_MAX_GB})",
                self.max_gb
            );
        }
        crate::config::check_range(
            "collect.max_age_days",
            self.max_age_days,
            0,
            MAX_MAX_AGE_DAYS,
        )?;
        crate::config::check_range(
            "collect.dedup_hamming",
            self.dedup_hamming,
            0,
            MAX_DEDUP_HAMMING,
        )?;
        validate_classes(&self.classes)?;
        for b in Bucket::ALL {
            let c = self.buckets.get(b);
            let field = format!("collect.buckets.{}", b.as_str());
            if !(c.rate.is_finite() && (0.0..=1.0).contains(&c.rate)) {
                bail!(
                    "{field}.rate: must be between 0 and 1 (the share of matching frames kept), \
                     got {}",
                    c.rate
                );
            }
            crate::config::check_range(
                &format!("{field}.per_camera_min_interval_s"),
                c.per_camera_min_interval_s,
                0,
                MAX_INTERVAL_S,
            )?;
            match (b, c.margin) {
                (Bucket::NearThreshold, Some(m)) => {
                    if !(m.is_finite() && m > 0.0 && m <= MAX_NEAR_MARGIN) {
                        bail!(
                            "{field}.margin: must be above 0 and at most {MAX_NEAR_MARGIN}, \
                             got {m}"
                        );
                    }
                }
                (Bucket::NearThreshold, None) => {}
                (_, Some(_)) => {
                    bail!("{field}.margin: only near_threshold has a margin; remove it here")
                }
                (_, None) => {}
            }
        }
        Ok(())
    }
}

/// `collect.classes`: 1 to [`MAX_CLASSES`] unique names of at most [`MAX_CLASS_LEN`]
/// characters, no control characters, no leading or trailing spaces.
pub fn validate_classes(classes: &[String]) -> anyhow::Result<()> {
    use anyhow::bail;
    if classes.is_empty() || classes.len() > MAX_CLASSES {
        bail!(
            "collect.classes: give 1 to {MAX_CLASSES} class names (the Review page's class list), \
             got {}",
            classes.len()
        );
    }
    let mut seen = std::collections::BTreeSet::new();
    for c in classes {
        if c.is_empty()
            || c.chars().count() > MAX_CLASS_LEN
            || c.trim() != c
            || c.chars().any(char::is_control)
        {
            bail!(
                "collect.classes: '{}' is not a valid class name (1 to {MAX_CLASS_LEN} characters, \
                 no control characters, no leading or trailing spaces)",
                c.escape_debug()
            );
        }
        if !seen.insert(c.as_str()) {
            bail!("collect.classes: '{c}' is listed twice; class names must be unique");
        }
    }
    Ok(())
}

/// The resolved settings the workers carry ([`crate::worker::LiveSettings::collect`]) and the
/// collector thread applies per frame.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectSettings {
    pub enabled: bool,
    /// Absolute dataset folder.
    pub dir: PathBuf,
    pub buckets: Buckets,
    pub max_bytes: u64,
    pub max_age_days: u32,
    pub dedup_hamming: u32,
}

impl Default for CollectSettings {
    fn default() -> Self {
        Self::from_config(&CollectConfig::default(), &crate::exe_dir())
    }
}

impl CollectSettings {
    pub fn from_config(c: &CollectConfig, data_root: &Path) -> Self {
        Self {
            enabled: c.enabled,
            dir: c.resolved_dir(data_root),
            buckets: c.buckets.clone(),
            max_bytes: c.max_bytes(),
            max_age_days: c.max_age_days,
            dedup_hamming: c.dedup_hamming,
        }
    }

    /// The settings of `config` (its `collect` section under its data root).
    pub fn for_config(config: &crate::config::Config) -> Self {
        Self::from_config(&config.collect, &config.data_root())
    }

    /// Below-threshold detections are worth computing: collection and `near_threshold` on.
    pub fn wants_below_threshold(&self) -> bool {
        self.enabled && self.buckets.near_threshold.enabled
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let c = CollectConfig::default();
        assert!(!c.enabled);
        assert!(c.is_default());
        assert_eq!(c.max_gb, 5.0);
        assert_eq!(c.max_age_days, 90);
        assert_eq!(c.dedup_hamming, 6);
        assert_eq!(c.classes.len(), 14);
        assert_eq!(c.classes[7], "raccoon");
        assert_eq!(c.buckets.empty.per_camera_min_interval_s, 600);
        assert_eq!(c.buckets.near_margin(), 0.1);
        assert!(Bucket::ALL.iter().all(|b| c.buckets.get(*b).enabled));
        c.validate(Path::new("/data")).unwrap();
        assert_eq!(
            c.resolved_dir(Path::new("/data")),
            Path::new("/data").join("dataset")
        );
        assert_eq!(
            c.resolved_review_dir(Path::new("/data")),
            Path::new("/data").join("dataset")
        );
        let r = CollectConfig {
            review_dir: Some("imports/drive".into()),
            ..CollectConfig::default()
        };
        assert_eq!(
            r.resolved_review_dir(Path::new("/data")),
            Path::new("/data").join("imports/drive")
        );
        assert!(!r.is_default());
        assert!(
            !serde_json::to_string(&CollectConfig::default())
                .unwrap()
                .contains("review_dir")
        );
        assert_eq!(c.max_bytes(), 5 * 1024 * 1024 * 1024);
    }

    #[test]
    fn partial_buckets_keep_their_own_defaults() {
        let c: CollectConfig = serde_json::from_str(
            r#"{"enabled": true, "buckets": {"empty": {"enabled": false}, "near_threshold": {"margin": 0.05}}}"#,
        )
        .unwrap();
        assert!(c.enabled);
        assert!(!c.buckets.empty.enabled);
        assert_eq!(c.buckets.empty.per_camera_min_interval_s, 600);
        assert_eq!(c.buckets.near_margin(), 0.05);
        assert_eq!(
            c.buckets.positive,
            BucketConfig::default_for(Bucket::Positive)
        );
        c.validate(Path::new("/data")).unwrap();
    }

    #[test]
    fn round_trip_with_clean_numbers() {
        let mut c = CollectConfig {
            enabled: true,
            dir: Some("frames".into()),
            max_gb: 0.5,
            max_age_days: 30,
            dedup_hamming: 4,
            ..CollectConfig::default()
        };
        c.buckets.positive.rate = 0.3;
        c.buckets.near_threshold.margin = Some(0.15);
        let text = serde_json::to_string(&c).unwrap();
        assert!(text.contains("\"rate\":0.3,"), "{text}");
        assert!(text.contains("\"margin\":0.15"), "{text}");
        assert!(text.contains("\"max_gb\":0.5"), "{text}");
        let back: CollectConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        assert_eq!(
            back.resolved_dir(Path::new("/data")),
            Path::new("/data").join("frames")
        );
    }

    #[test]
    fn validation_names_the_field() {
        let root = Path::new("/data");
        let bad = |edit: &dyn Fn(&mut CollectConfig), needle: &str| {
            let mut c = CollectConfig::default();
            edit(&mut c);
            let e = c.validate(root).unwrap_err().to_string();
            assert!(e.contains(needle), "{e} lacks {needle}");
        };
        bad(&|c| c.max_gb = 0.0, "collect.max_gb");
        bad(&|c| c.max_gb = f32::NAN, "collect.max_gb");
        bad(&|c| c.max_age_days = 5000, "collect.max_age_days");
        bad(&|c| c.dedup_hamming = 33, "collect.dedup_hamming");
        bad(
            &|c| c.buckets.empty.rate = 1.5,
            "collect.buckets.empty.rate",
        );
        bad(
            &|c| c.buckets.suppressed.per_camera_min_interval_s = MAX_INTERVAL_S + 1,
            "collect.buckets.suppressed.per_camera_min_interval_s",
        );
        bad(
            &|c| c.buckets.near_threshold.margin = Some(0.0),
            "collect.buckets.near_threshold.margin",
        );
        bad(
            &|c| c.buckets.positive.margin = Some(0.1),
            "only near_threshold has a margin",
        );
        bad(&|c| c.dir = Some("../x".into()), "collect.dir");
        bad(&|c| c.dir = Some("/etc/frames".into()), "system directory");
        bad(&|c| c.classes.clear(), "collect.classes");
        bad(&|c| c.classes.push("person".into()), "listed twice");
        bad(&|c| c.classes.push(" cow".into()), "not a valid class name");
        bad(&|c| c.classes.push("a\nb".into()), "not a valid class name");
    }

    #[test]
    fn config_round_trip_and_validation() {
        let mut c = crate::config::Config::default();
        assert!(
            !c.saved_text().unwrap().contains("collect"),
            "default not written"
        );
        c.collect.enabled = true;
        c.collect.buckets.empty.per_camera_min_interval_s = 300;
        c.collect.buckets.positive.rate = 0.25;
        let text = c.saved_text().unwrap();
        assert!(text.contains("\"collect\""), "{text}");
        let back: crate::config::Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        assert_eq!(back.saved_text().unwrap(), text);
        back.validate().unwrap();
        c.collect.dedup_hamming = 99;
        let e = c.validate().unwrap_err().to_string();
        assert!(e.contains("collect.dedup_hamming"), "{e}");
    }

    /// Every key of the section has a README row (`| `collect.<key>` |`).
    #[test]
    fn readme_documents_every_collect_key() {
        let readme = include_str!("../../README.md");
        let c = CollectConfig {
            dir: Some("d".into()),
            ..CollectConfig::default()
        };
        let serde_json::Value::Object(m) = serde_json::to_value(&c).unwrap() else {
            panic!("not an object")
        };
        assert!(readme.contains("| `collect` |"));
        for k in m.keys() {
            assert!(
                readme.contains(&format!("| `collect.{k}` |")),
                "README lacks collect.{k}"
            );
        }
        for b in Bucket::ALL {
            assert!(
                readme.contains(&format!("`{}`", b.as_str())),
                "README lacks {b:?}"
            );
        }
    }

    #[test]
    fn unknown_keys_are_refused() {
        assert!(serde_json::from_str::<CollectConfig>(r#"{"enable": true}"#).is_err());
        assert!(
            serde_json::from_str::<CollectConfig>(r#"{"buckets": {"positives": {}}}"#).is_err()
        );
        assert!(
            serde_json::from_str::<CollectConfig>(r#"{"buckets": {"empty": {"interval": 5}}}"#)
                .is_err()
        );
    }
}
