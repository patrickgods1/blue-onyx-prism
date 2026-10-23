//! Service configuration: a JSON file next to the executable, merged with CLI flags.
//! Merge rule: a flag given on the command line overrides the file value (`cli::merge_cli`; an
//! absent flag leaves the file value alone), the result is validated and written back to the
//! file, except the one-off flags (`--allow-remote-admin`, `--force-cpu`, `--save-ref-image`),
//! which apply to that run only and are never written back (`cli::without_one_off_flags`).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::model::ModelFamilyKind;
use crate::model::enhance::{Preprocess, PreprocessModes};
use crate::model::filters::OverlapRule;
use crate::static_memory::StaticSuppression;

pub const CONFIG_FILE: &str = "blue_onyx_prism_config.json";
pub const SERVICE_CONFIG_FILE: &str = "blue_onyx_prism_config_service.json";
/// Config file names used before the project was renamed from Blue Onyx OpenVINO.
const LEGACY_CONFIG_FILE: &str = "blue_onyx_openvino_config.json";
const LEGACY_SERVICE_CONFIG_FILE: &str = "blue_onyx_openvino_config_service.json";

/// Rename `dir/legacy` to `dir/current` when only the legacy file exists. Returns the path to use.
fn migrate_legacy(dir: &Path, current: &str, legacy: &str) -> PathBuf {
    let path = dir.join(current);
    let old = dir.join(legacy);
    if !path.exists() && old.is_file() {
        // Logged through `logbuf::early`: this runs before logging is set up (CO-11).
        match std::fs::rename(&old, &path) {
            Ok(()) => crate::logbuf::early(
                tracing::Level::INFO,
                format!(
                    "renamed the config file {} -> {} (the project's old name); nothing to do",
                    old.display(),
                    path.display()
                ),
            ),
            Err(e) => {
                crate::logbuf::early(
                    tracing::Level::WARN,
                    format!(
                        "could not rename the config file {} to {}: {e}. The folder is probably \
                         read-only for this account; the old file is used as is (rename it by \
                         hand to silence this warning)",
                        old.display(),
                        path.display()
                    ),
                );
                return old;
            }
        }
    }
    path
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// Format of the log files under `log_path` (`log_format`). The console always gets text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase", try_from = "String")]
pub enum LogFormat {
    /// Human-readable lines (the default).
    #[default]
    Text,
    /// One JSON object per line (timestamp, level, target, fields), for log shippers.
    Json,
}

impl LogFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogFormat::Text => "text",
            LogFormat::Json => "json",
        }
    }
}

impl TryFrom<String> for LogFormat {
    type Error = anyhow::Error;
    fn try_from(s: String) -> Result<Self> {
        s.parse()
    }
}

impl std::str::FromStr for LogFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "text" => Ok(LogFormat::Text),
            "json" => Ok(LogFormat::Json),
            other => anyhow::bail!("log_format: must be \"text\" or \"json\", got '{other}'"),
        }
    }
}

/// Time zone of log timestamps (`log_timezone`). Stored in the config as a string: `"utc"`
/// (the default), `"local"` (the computer's zone), an IANA name (`"America/Los_Angeles"`) or a
/// fixed offset (`"+08:00"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogTimezone {
    /// UTC, `2026-10-16T00:54:59.580Z` (the default).
    #[default]
    Utc,
    /// The computer's time zone.
    Local,
    /// An IANA zone (follows its daylight saving rules).
    Named(chrono_tz::Tz),
    /// A fixed offset from UTC.
    Fixed(chrono::FixedOffset),
}

impl LogTimezone {
    /// `t` as a timestamp: `...580Z` in UTC, else RFC 3339 with milliseconds and the numeric
    /// offset (`2026-10-15T17:54:59.580-07:00`).
    pub fn format(&self, t: chrono::DateTime<chrono::Utc>) -> String {
        use chrono::SecondsFormat::Millis;
        match self {
            LogTimezone::Utc => t.to_rfc3339_opts(Millis, true),
            LogTimezone::Local => t
                .with_timezone(&chrono::Local)
                .to_rfc3339_opts(Millis, false),
            LogTimezone::Named(tz) => t.with_timezone(tz).to_rfc3339_opts(Millis, false),
            LogTimezone::Fixed(o) => t.with_timezone(o).to_rfc3339_opts(Millis, false),
        }
    }
}

impl std::fmt::Display for LogTimezone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogTimezone::Utc => f.write_str("utc"),
            LogTimezone::Local => f.write_str("local"),
            LogTimezone::Named(tz) => f.write_str(tz.name()),
            LogTimezone::Fixed(o) => {
                let secs = o.local_minus_utc();
                let (sign, a) = if secs < 0 { ('-', -secs) } else { ('+', secs) };
                write!(f, "{sign}{:02}:{:02}", a / 3600, a % 3600 / 60)
            }
        }
    }
}

impl std::str::FromStr for LogTimezone {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let t = s.trim();
        let bad = || {
            anyhow::anyhow!(
                "log_timezone: '{t}' is not a time zone. Use \"utc\", \"local\" (this computer's                  zone), an IANA name such as \"America/Los_Angeles\" or a fixed offset such as                  \"+08:00\" or \"-05:30\"; fix the value in the config file or on the Config page"
            )
        };
        match t.to_ascii_lowercase().as_str() {
            "utc" => return Ok(LogTimezone::Utc),
            "local" => return Ok(LogTimezone::Local),
            _ => {}
        }
        if let Some(sign @ ('+' | '-')) = t.chars().next() {
            let digits = t.get(1..).unwrap_or_default().as_bytes();
            let &[h1, h2, b':', m1, m2] = digits else {
                return Err(bad());
            };
            if ![h1, h2, m1, m2].iter().all(u8::is_ascii_digit) {
                return Err(bad());
            }
            let two = |a: u8, b: u8| i32::from(a - b'0') * 10 + i32::from(b - b'0');
            let (h, m) = (two(h1, h2), two(m1, m2));
            if h > 23 || m > 59 {
                return Err(bad());
            }
            let secs = (h * 3600 + m * 60) * if sign == '-' { -1 } else { 1 };
            return chrono::FixedOffset::east_opt(secs)
                .map(LogTimezone::Fixed)
                .ok_or_else(bad);
        }
        t.parse::<chrono_tz::Tz>()
            .map(LogTimezone::Named)
            .map_err(|_| bad())
    }
}

impl TryFrom<String> for LogTimezone {
    type Error = anyhow::Error;
    fn try_from(s: String) -> Result<Self> {
        s.parse()
    }
}

impl Serialize for LogTimezone {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for LogTimezone {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl std::str::FromStr for LogLevel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "trace" => LogLevel::Trace,
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" | "warning" => LogLevel::Warn,
            "error" => LogLevel::Error,
            other => anyhow::bail!(
                "unknown log level '{other}'; use one of trace, debug, info, warn, error"
            ),
        })
    }
}

/// A confidence threshold rounded to 2 decimals (the precision the UI, the benchmark's
/// "rounded down to 0.01" picks and the config file use), as the nearest `f32`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "narrowing the f64 form of an f32 back to f32 is the point (nearest f32)"
)]
pub fn round_threshold(t: f32) -> f32 {
    round_threshold_f64(t) as f32
}

/// [`round_threshold`] as the `f64` to put in JSON: `0.35`, not `0.3499999940395355` (what an
/// `f32` widened to `f64` prints).
pub fn round_threshold_f64(t: f32) -> f64 {
    if !t.is_finite() {
        return t as f64;
    }
    (t as f64 * 100.0).round() / 100.0
}

/// An `f32` as the `f64` with the same shortest decimal form (`0.45f32` -> `0.45`), so values
/// that pass through `serde_json::Value` (which stores `f64`) print as typed.
pub fn clean_f32(x: f32) -> f64 {
    if !x.is_finite() {
        return x as f64;
    }
    format!("{x}").parse::<f64>().unwrap_or(x as f64)
}

/// Serializers for threshold fields (see [`round_threshold_f64`] and [`clean_f32`]).
pub mod thr {
    use serde::Serializer;
    pub fn ser<S: Serializer>(t: &f32, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(super::round_threshold_f64(*t))
    }
    pub fn ser_opt<S: Serializer>(t: &Option<f32>, s: S) -> Result<S::Ok, S::Error> {
        match t {
            Some(t) => s.serialize_some(&super::round_threshold_f64(*t)),
            None => s.serialize_none(),
        }
    }
    pub fn ser_clean<S: Serializer>(t: &f32, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(super::clean_f32(*t))
    }
    pub fn ser_opt_clean<S: Serializer>(t: &Option<f32>, s: S) -> Result<S::Ok, S::Error> {
        match t {
            Some(t) => s.serialize_some(&super::clean_f32(*t)),
            None => s.serialize_none(),
        }
    }
    pub fn ser_opt_pair_clean<S: Serializer>(
        t: &Option<[f32; 2]>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match t {
            Some([a, b]) => s.serialize_some(&[super::clean_f32(*a), super::clean_f32(*b)]),
            None => s.serialize_none(),
        }
    }
    fn clean_map(m: &super::BTreeMap<String, f32>) -> super::BTreeMap<&str, f64> {
        m.iter()
            .map(|(k, v)| (k.as_str(), super::clean_f32(*v)))
            .collect()
    }
    pub fn ser_map_clean<S: Serializer>(
        m: &super::BTreeMap<String, f32>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&clean_map(m), s)
    }
    pub fn ser_opt_map_clean<S: Serializer>(
        m: &Option<super::BTreeMap<String, f32>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match m {
            Some(m) => s.serialize_some(&clean_map(m)),
            None => s.serialize_none(),
        }
    }
}

/// One model entry. Relative paths resolve against the executable directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    /// Name used in `/v1/vision/custom/{name}`; defaults to the file stem.
    pub name: Option<String>,
    /// `.xml` (OpenVINO IR, with `.bin` beside it) or `.onnx`.
    pub path: PathBuf,
    pub family: ModelFamilyKind,
    /// YAML with a `NAMES:` list. Defaults to `<stem>.yaml` next to the model, then COCO-80.
    pub classes: Option<PathBuf>,
    /// Per-model device override, same syntax as the global `device` ("GPU", "openvino:cpu",
    /// ...). None = global setting.
    pub device: Option<String>,
    /// Per-model confidence threshold (None = global). Kept rounded to 2 decimals.
    #[serde(serialize_with = "thr::ser_opt")]
    pub confidence_threshold: Option<f32>,
    pub object_filter: Option<Vec<String>>,
    /// Compile on first request instead of at startup. That request (and every one until the
    /// model is ready) is answered at once with "Model 'x' is initializing", never queued.
    pub lazy: bool,
    /// Per-model inference precision hint on GPU ("f16" default, "f32" escape hatch).
    pub gpu_precision: Option<String>,
    /// Load this model at startup. Disabled models stay in the config but are not loaded or
    /// served.
    pub enabled: bool,
    /// Concurrent inferences of this model (None = the global `inflight`). Restart.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inflight: Option<usize>,
    /// Night / IR false-positive rules ([`crate::model::filters`]); None = the global value.
    /// Maps (`label_thresholds`, `night_label_thresholds`) are merged over the global ones
    /// (this model's entries win); the other fields replace the global value. Applies at once.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "thr::ser_opt_clean"
    )]
    pub night_confidence_threshold: Option<f32>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "thr::ser_opt_map_clean"
    )]
    pub label_thresholds: Option<BTreeMap<String, f32>>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "thr::ser_opt_map_clean"
    )]
    pub night_label_thresholds: Option<BTreeMap<String, f32>>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "thr::ser_opt_pair_clean"
    )]
    pub person_aspect: Option<[f32; 2]>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "thr::ser_opt_clean"
    )]
    pub min_box_area_frac: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suppress_overlaps: Option<Vec<OverlapRule>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_suppression: Option<StaticSuppression>,
    /// Image enhancement before the model's resize ([`crate::model::enhance`]); None = the
    /// global `preprocess`. Applies at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preprocess: Option<Preprocess>,
    /// Enhancement on night / IR frames, replacing `preprocess` there; None = the global
    /// `night_preprocess`. Applies at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub night_preprocess: Option<Preprocess>,
    /// Another configured model that answers this model's night / IR frames (scene `night` or
    /// `ir`, see [`crate::image_stats`]): the decoded frame is forwarded to its worker and its
    /// result returned. It must not have a `night_model` itself. Falls back to this model when
    /// the night model is not served or not ready. Applies at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub night_model: Option<String>,
    /// Verifier cascade ([`crate::verify`]): another configured model checks this model's
    /// candidate boxes; None = the global `verify`. Applies at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verify: Option<crate::verify::VerifyConfig>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: None,
            path: PathBuf::new(),
            family: ModelFamilyKind::Auto,
            classes: None,
            device: None,
            confidence_threshold: None,
            object_filter: None,
            lazy: false,
            gpu_precision: None,
            enabled: true,
            inflight: None,
            night_confidence_threshold: None,
            label_thresholds: None,
            night_label_thresholds: None,
            person_aspect: None,
            min_box_area_frac: None,
            suppress_overlaps: None,
            static_suppression: None,
            preprocess: None,
            night_preprocess: None,
            night_model: None,
            verify: None,
        }
    }
}

impl Config {
    /// The enhancement modes of model `m`: its own `preprocess` / `night_preprocess` over the
    /// global ones (each field separately; an unset night mode means `preprocess`).
    pub fn preprocess_modes(&self, m: &ModelConfig) -> PreprocessModes {
        PreprocessModes {
            preprocess: m.preprocess.or(self.preprocess).unwrap_or_default(),
            night_preprocess: m.night_preprocess.or(self.night_preprocess),
        }
    }
}

impl ModelConfig {
    pub fn effective_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| {
            self.path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".to_string())
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub port: u16,
    /// Address the HTTP listener binds (`0.0.0.0` = every IPv4 interface, plus `[::1]` for
    /// `localhost`). `127.0.0.1` keeps the service local to this machine. Process restart.
    pub bind_address: std::net::IpAddr,
    /// Serve the admin pages and actions (Config, Test, Logs, Benchmark, resources, restart)
    /// to other machines too. Off: only loopback clients get them (403 otherwise); detection,
    /// `/stats.json`, `/prometheus`, `/v1/devices` and `/v1/config` stay open. Applies at once.
    pub allow_remote_admin: bool,
    /// How long a request may wait in the queue + processing before it is dropped (the only
    /// request timeout; the HTTP handler waits this plus a 500 ms grace for the worker).
    pub request_timeout_secs: u64,
    /// 0 = auto-size from timeout and measured inference time.
    pub worker_queue_size: usize,
    /// Global inference device spec (`backend::spec`): "auto" (default: best runtime and device
    /// for the hardware), "openvino:gpu.1", "ort:cuda", "ort:coreml", ... Legacy "GPU", "GPU.N"
    /// and "CPU" keep meaning OpenVINO.
    pub device: String,
    pub gpu_index: u32,
    /// Use the best CPU option: OpenVINO CPU, or ONNX Runtime CPU when OpenVINO is missing.
    pub force_cpu: bool,
    /// Compiled-model cache directory (relative to exe dir). Empty disables caching.
    pub cache_dir: String,
    /// Directory holding the OpenVINO runtime (archive layout). None = `<exe_dir>/openvino` or system.
    pub openvino_dir: Option<PathBuf>,
    /// Directory holding the ONNX Runtime library (or the library file itself). None =
    /// `ORT_DYLIB_PATH`, then `<exe_dir>/onnxruntime`.
    pub onnxruntime_dir: Option<PathBuf>,
    /// Kept rounded to 2 decimals ([`round_threshold`]).
    #[serde(serialize_with = "thr::ser")]
    pub confidence_threshold: f32,
    #[serde(serialize_with = "thr::ser_clean")]
    pub nms_iou: f32,
    /// Only report these labels (case-insensitive). Empty = all.
    pub object_filter: Vec<String>,
    /// Minimum score on night / IR frames ([`crate::image_stats::Scene`]); raises, never lowers,
    /// the threshold. None = off. Applies at once.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "thr::ser_opt_clean"
    )]
    pub night_confidence_threshold: Option<f32>,
    /// Minimum score per label (`{"person": 0.6}`, case-insensitive). Applies at once.
    #[serde(
        skip_serializing_if = "BTreeMap::is_empty",
        serialize_with = "thr::ser_map_clean"
    )]
    pub label_thresholds: BTreeMap<String, f32>,
    /// Minimum score per label on night / IR frames. Applies at once.
    #[serde(
        skip_serializing_if = "BTreeMap::is_empty",
        serialize_with = "thr::ser_map_clean"
    )]
    pub night_label_thresholds: BTreeMap<String, f32>,
    /// Allowed width / height of a `person` box, `[min, max]` (suggested `[0.2, 0.9]`). None =
    /// off. Applies at once.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "thr::ser_opt_pair_clean"
    )]
    pub person_aspect: Option<[f32; 2]>,
    /// Drop boxes smaller than this share of the image area. None = off. Applies at once.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "thr::ser_opt_clean"
    )]
    pub min_box_area_frac: Option<f32>,
    /// Drop a box of one label that overlaps a box of another (`person` on `potted plant`).
    /// Applies at once.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suppress_overlaps: Vec<OverlapRule>,
    /// Suppress boxes that repeat at the same place ([`crate::static_memory`]). Off by
    /// default. Applies at once.
    #[serde(skip_serializing_if = "StaticSuppression::is_default")]
    pub static_suppression: StaticSuppression,
    /// Image enhancement before the model's resize: `none`, `grayscale`, `gamma:<g>`,
    /// `clahe[:<clip>[,<tiles>]]` ([`crate::model::enhance`]). None = off. Applies at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preprocess: Option<Preprocess>,
    /// Enhancement on night / IR frames, replacing `preprocess` there. None = `preprocess`.
    /// Applies at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub night_preprocess: Option<Preprocess>,
    /// Verifier cascade for every model without its own `verify` ([`crate::verify`]); the
    /// verifier model itself is never verified. None = off. Applies at once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verify: Option<crate::verify::VerifyConfig>,
    /// Dataset collector ([`crate::collect`], `docs/DATASET.md`): saves frames with what the
    /// model saw for review and training. Off by default. Applies at once.
    #[serde(skip_serializing_if = "crate::collect::CollectConfig::is_default")]
    pub collect: crate::collect::CollectConfig,
    pub log_level: LogLevel,
    /// Directory of the daily rolling log files (None = stdout only). Relative paths resolve
    /// against the exe dir. Process restart.
    pub log_path: Option<PathBuf>,
    /// Format of the log files under `log_path`: `"text"` (default) or `"json"` (one object
    /// per line, for log shippers); the console stays text. An unknown value is a parse error
    /// naming the key. Process restart.
    pub log_format: LogFormat,
    /// Time zone of log timestamps: `"utc"` (default), `"local"`, an IANA name
    /// (`"America/Los_Angeles"`) or a fixed offset (`"+08:00"`). An unknown value is a parse
    /// error naming the key. Applies at once.
    pub log_timezone: LogTimezone,
    /// Save annotated images here (optional).
    pub save_image_path: Option<PathBuf>,
    pub save_ref_image: bool,
    /// Delete saved images older than this many days (0 = keep forever). Applies at once.
    pub save_image_max_age_days: u32,
    /// Keep the saved-image folder under this many MB, oldest files deleted first
    /// (0 = unlimited). Applies at once.
    pub save_image_max_mb: u64,
    /// CPU inference threads (0 = all physical cores but one on machines with at least 4 cores,
    /// every core on smaller ones; the OS and the desktop keep a core; see
    /// [`crate::backend::cpu_threads`]). Restart.
    pub intra_threads: usize,
    /// Decode + preprocess threads shared by every model (0 = from the hardware profile:
    /// `clamp(cores / 2, 1, 8)`, at most 2 when every model runs on the CPU). Restart.
    pub preprocess_threads: usize,
    /// Concurrent inferences per model (0 = from the hardware profile and the model's device,
    /// see [`crate::profile::Profile::tuning_for`]); a model's own `inflight` wins. Restart.
    pub inflight: usize,
    /// ONNX Runtime sessions per model, one inference in flight each (0 = from the hardware
    /// profile: 2 on a machine with a discrete GPU of at least 4 GB and 8 GB of memory, else 1).
    /// Each session holds its own copy of the model, one inference in flight each. Restart.
    pub ort_sessions: usize,
    /// `auto` (classify the hardware) or a forced class: `low`, `mid`, `high`. Restart.
    pub hardware_profile: crate::profile::HardwareProfile,
    pub models_dir: PathBuf,
    /// Download missing resources (runtimes, models) in the background at startup. `false`
    /// (air-gapped, Docker) only reports what is missing and the command to fetch it.
    pub auto_download: bool,
    /// Allow downloads over 500 MB (`resources::LARGE_DOWNLOAD_BYTES`), in practice the NVIDIA
    /// CUDA/cuDNN libraries. Off by default.
    pub allow_large_downloads: bool,
    /// Root for downloaded runtimes (`openvino/`, `onnxruntime/<flavor>/`, ...), relative to the
    /// exe dir. None = the exe dir. Model files go where each model's `path` points.
    pub download_dir: Option<PathBuf>,
    /// Largest image accepted for decoding, in pixels (width x height), checked from the image
    /// header before any pixel buffer is allocated; applies to detection requests, the test
    /// page, saved images and benchmark datasets. Applies at once.
    pub max_image_pixels: u64,
    /// Name of the model that serves `/v1/vision/detection`. None = first enabled entry.
    pub default_model: Option<String>,
    pub models: Vec<ModelConfig>,
    /// Defaults of the benchmark (web UI Benchmark page and `benchmark` CLI).
    pub benchmark: BenchmarkConfig,
}

/// A benchmark dataset: a built-in set id (`coco-cctv`), `sample` (the embedded image),
/// `dir:<path>`, or a directory object with optional ground truth and display name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DatasetRef {
    Id(String),
    Dir {
        dir: PathBuf,
        /// COCO JSON file or YOLO labels directory (default: found in `dir`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gt: Option<PathBuf>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

impl DatasetRef {
    /// The directory of a `dir:<path>` id or a directory object.
    pub fn dir(&self) -> Option<(PathBuf, Option<PathBuf>, Option<String>)> {
        match self {
            DatasetRef::Id(s) => s
                .trim()
                .strip_prefix(crate::benchmark::images::DIR_PREFIX)
                .map(|d| (PathBuf::from(d.trim()), None, None)),
            DatasetRef::Dir { dir, gt, name } => Some((dir.clone(), gt.clone(), name.clone())),
        }
    }

    /// Display / CLI form: the id, or `dir:<path>`.
    pub fn label(&self) -> String {
        match self {
            DatasetRef::Id(s) => s.trim().to_string(),
            DatasetRef::Dir { dir, name, .. } => match name {
                Some(n) if !n.trim().is_empty() => n.trim().to_string(),
                _ => format!("{}{}", crate::benchmark::images::DIR_PREFIX, dir.display()),
            },
        }
    }
}

/// Benchmark defaults (`benchmark` in the config file). Old config files without it get these
/// defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchmarkConfig {
    /// Datasets to run on (see [`DatasetRef`]). Default: the three built-in sets.
    pub datasets: Vec<DatasetRef>,
    /// Images per dataset, evenly spread (0 = all).
    pub max_images_per_dataset: usize,
    /// Device specs; empty = every runnable option.
    pub devices: Vec<String>,
    /// Model names; empty = the enabled models whose files are present.
    pub models: Vec<String>,
    /// Untimed warm-up iterations per model and device.
    pub warmup: usize,
    /// Timed runs per image.
    pub repeat_per_image: usize,
    /// Model whose detections are the pseudo ground truth of datasets without ground truth;
    /// None = the most accurate installed model.
    pub reference_model: Option<String>,
    /// Accuracy vs speed weights of the overall grade.
    pub weights: crate::benchmark::grade::Weights,
    /// Download the selected built-in datasets when a run starts (large ones still need
    /// `allow_large_downloads`).
    pub auto_download_datasets: bool,
    /// What the best confidence threshold optimizes: `f1` (default), `f2` (favor recall),
    /// `precision:<p>` (highest recall with precision >= p), `recall:<r>`, `youden` (frame-level
    /// TPR - FPR) or `fpr:<x>` (highest frame TPR with frame FPR <= x).
    pub threshold_objective: crate::benchmark::threshold::Objective,
    /// What the accuracy grade, the ranking and the device recommendation use: `ap50` (default,
    /// box-level AP@0.5 with small-object recall) or `roc_auc` (macro frame-level ROC AUC).
    pub accuracy_metric: crate::benchmark::grade::AccuracyMetric,
    /// Run benchmarks at below-normal process priority so the desktop stays responsive
    /// (Windows: the whole service for the run, restored after; CLI on Unix: nice 10). Applies
    /// at once.
    pub low_priority: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            datasets: crate::benchmark::images::DEFAULT_SET_IDS
                .iter()
                .map(|s| DatasetRef::Id(s.to_string()))
                .collect(),
            max_images_per_dataset: 0,
            devices: Vec::new(),
            models: Vec::new(),
            warmup: 3,
            repeat_per_image: 1,
            reference_model: None,
            weights: Default::default(),
            auto_download_datasets: true,
            threshold_objective: Default::default(),
            accuracy_metric: Default::default(),
            low_priority: true,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: crate::DEFAULT_PORT,
            bind_address: std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
            allow_remote_admin: false,
            request_timeout_secs: 15,
            worker_queue_size: 0,
            device: "auto".to_string(),
            gpu_index: 0,
            force_cpu: false,
            cache_dir: "cache".to_string(),
            openvino_dir: None,
            onnxruntime_dir: None,
            confidence_threshold: 0.5,
            nms_iou: 0.5,
            object_filter: Vec::new(),
            night_confidence_threshold: None,
            label_thresholds: BTreeMap::new(),
            night_label_thresholds: BTreeMap::new(),
            person_aspect: None,
            min_box_area_frac: None,
            suppress_overlaps: Vec::new(),
            static_suppression: StaticSuppression::default(),
            preprocess: None,
            night_preprocess: None,
            verify: None,
            collect: crate::collect::CollectConfig::default(),
            log_level: LogLevel::Info,
            log_path: None,
            log_format: LogFormat::Text,
            log_timezone: LogTimezone::Utc,
            save_image_path: None,
            save_ref_image: false,
            save_image_max_age_days: DEFAULT_SAVE_IMAGE_MAX_AGE_DAYS,
            save_image_max_mb: DEFAULT_SAVE_IMAGE_MAX_MB,
            intra_threads: 0,
            preprocess_threads: 0,
            inflight: 0,
            ort_sessions: 0,
            hardware_profile: crate::profile::HardwareProfile::Auto,
            models_dir: PathBuf::from("models"),
            auto_download: true,
            allow_large_downloads: false,
            download_dir: None,
            max_image_pixels: crate::image::MAX_IMAGE_PIXELS_DEFAULT,
            default_model: None,
            models: Vec::new(),
            benchmark: BenchmarkConfig::default(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| {
            format!(
                "could not read the config file {} (missing permission or a locked file?); \
                 check the file, or pass --config <another file>",
                path.display()
            )
        })?;
        let mut config: Config = serde_json::from_str(&text).with_context(|| {
            format!(
                "the config file {} is not valid config JSON (most likely a hand edit: a missing \
                 comma or quote, or a misspelled value); fix the place named below or restore \
                 a backup",
                path.display()
            )
        })?;
        config.normalize();
        config.validate().with_context(|| {
            format!(
                "the config file {} has a value out of range (most likely a hand edit); fix the \
                 key named below or delete it to use its default",
                path.display()
            )
        })?;
        Ok(config)
    }

    /// Give a config without any model the default catalog model ([`DEFAULT_MODEL_NAME`],
    /// `<models_dir>/IPcam-general.onnx`, YOLOv5), so a first start serves the web UI and
    /// `auto_download` fetches the file (CO-05). Returns the name of the added model, or None
    /// when the config already has models.
    pub fn seed_default_model(&mut self) -> Option<String> {
        if !self.models.is_empty() {
            return None;
        }
        let family = crate::download::find(DEFAULT_MODEL_NAME)
            .and_then(|r| match r.provides {
                crate::resources::catalog::Provides::Model { family, .. } => Some(family),
                _ => None,
            })
            .unwrap_or(ModelFamilyKind::Yolo5);
        self.models.push(ModelConfig {
            name: Some(DEFAULT_MODEL_NAME.to_string()),
            path: self.models_dir.join(format!("{DEFAULT_MODEL_NAME}.onnx")),
            family,
            ..Default::default()
        });
        Some(DEFAULT_MODEL_NAME.to_string())
    }

    /// The text [`Self::save`] writes: pretty JSON with every model entry named
    /// ([`ModelConfig::effective_name`]).
    pub fn saved_text(&self) -> Result<String> {
        let mut named = self.clone();
        named.fill_model_names();
        Ok(serde_json::to_string_pretty(&named)?)
    }

    /// Write the config as pretty JSON ([`Self::saved_text`]), atomically: the text goes to
    /// `<path>.tmp` in the same directory, is flushed to disk, then renamed over `path`
    /// (`std::fs::rename` replaces the target on Windows too). A crash or a full disk leaves
    /// the previous file intact; on failure the temp file is removed.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = self.saved_text()?;
        let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
        tmp_name.push(".tmp");
        let tmp = path.with_file_name(tmp_name);
        let write = || -> std::io::Result<()> {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(text.as_bytes())?;
            f.sync_all()?;
            drop(f);
            std::fs::rename(&tmp, path)
        };
        write().map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            anyhow::Error::new(e).context(format!("writing config {}", path.display()))
        })
    }

    /// `<exe_dir>/blue_onyx_prism_config.json`, adopting a pre-rename config file if present.
    pub fn default_config_path() -> PathBuf {
        migrate_legacy(&crate::exe_dir(), CONFIG_FILE, LEGACY_CONFIG_FILE)
    }

    pub fn service_config_path() -> PathBuf {
        migrate_legacy(
            &crate::exe_dir(),
            SERVICE_CONFIG_FILE,
            LEGACY_SERVICE_CONFIG_FILE,
        )
    }

    /// The in-memory form of a config as it is saved and loaded: every model named
    /// ([`Self::fill_model_names`]) and the confidence thresholds rounded to 2 decimals
    /// ([`round_threshold`]). The config store keeps configs normalized so a reloaded file
    /// compares equal to what was saved.
    pub fn normalize(&mut self) {
        self.fill_model_names();
        self.confidence_threshold = round_threshold(self.confidence_threshold);
        for m in &mut self.models {
            m.confidence_threshold = m.confidence_threshold.map(round_threshold);
        }
    }

    /// Give every model entry without a `name` its [`ModelConfig::effective_name`] (the file
    /// stem), so the JSON and every model list show it. Names that are set are kept.
    pub fn fill_model_names(&mut self) {
        for m in &mut self.models {
            if m.name.as_deref().is_none_or(|n| n.trim().is_empty()) {
                m.name = None;
                m.name = Some(m.effective_name());
            }
        }
    }

    /// The model that serves `/v1/vision/detection`.
    pub fn default_model_index(&self) -> Option<usize> {
        match &self.default_model {
            Some(name) => self.models.iter().position(|m| {
                crate::registry::normalize_name(&m.effective_name())
                    == crate::registry::normalize_name(name)
            }),
            None => (!self.models.is_empty()).then_some(0),
        }
    }

    /// Models that are loaded at startup (`enabled`), in config order.
    pub fn enabled_models(&self) -> impl Iterator<Item = &ModelConfig> {
        self.models.iter().filter(|m| m.enabled)
    }

    /// Index into `models` of the model that will actually serve `/v1/vision/detection`:
    /// `default_model` when it names an enabled model, else the first enabled model.
    pub fn effective_default_index(&self) -> Option<usize> {
        self.default_model_index()
            .filter(|&i| self.models.get(i).is_some_and(|m| m.enabled))
            .or_else(|| self.models.iter().position(|m| m.enabled))
    }

    /// Append `model` unless an entry with the same normalized name or the same path already
    /// exists. Returns whether it was added.
    pub fn add_model_if_absent(&mut self, model: ModelConfig) -> bool {
        let norm = |m: &ModelConfig| crate::registry::normalize_name(&m.effective_name());
        let new_name = norm(&model);
        let new_path = crate::resolve_path(&model.path);
        let exists = self
            .models
            .iter()
            .any(|m| norm(m) == new_name || crate::resolve_path(&m.path) == new_path);
        if !exists {
            self.models.push(model);
        }
        !exists
    }

    pub fn request_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.request_timeout_secs.max(1))
    }

    /// Effective device string for a model, honoring `force_cpu`, per-model override and
    /// `gpu_index`. This is the string reported as the requested device; see
    /// [`Self::device_spec_for`] for the parsed form.
    ///
    /// `gpu_index > 0` fills in the index of a global GPU-like spec without one (legacy `GPU`
    /// becomes `GPU.N`, `ort:cuda` becomes `ort:cuda:N`); a per-model override is used verbatim.
    pub fn device_for(&self, m: &ModelConfig) -> String {
        if self.force_cpu {
            return "CPU".to_string();
        }
        if let Some(d) = &m.device {
            return d.clone();
        }
        if self.gpu_index > 0 {
            if self.device.trim().eq_ignore_ascii_case("gpu") {
                return format!("GPU.{}", self.gpu_index);
            }
            if let Ok(spec) = crate::backend::spec::parse(&self.device) {
                let indexed = spec.with_default_index(self.gpu_index);
                if indexed != spec {
                    return indexed.to_string();
                }
            }
        }
        self.device.clone()
    }

    /// Parsed device for a model: `force_cpu` -> `openvino:cpu`, else the per-model override,
    /// else the global device with `gpu_index` applied (see [`Self::device_for`]). The registry
    /// turns `force_cpu` into the best CPU option of the machine (`Runtimes::best_cpu`), which is
    /// `ort:cpu` when OpenVINO is missing.
    pub fn device_spec_for(&self, m: &ModelConfig) -> Result<crate::backend::DeviceSpec> {
        if self.force_cpu {
            return Ok(crate::backend::DeviceSpec::OPENVINO_CPU);
        }
        let field = if m.device.is_some() {
            format!("model '{}': device", m.effective_name())
        } else {
            "device".to_string()
        };
        crate::backend::spec::parse(&self.device_for(m)).with_context(|| field)
    }

    /// Root for downloaded runtimes and relative model paths: `download_dir` (resolved against
    /// the exe dir), else the exe dir.
    pub fn data_root(&self) -> PathBuf {
        self.download_dir
            .as_deref()
            .map(crate::resolve_path)
            .unwrap_or_else(crate::exe_dir)
    }

    /// A model/classes/models_dir path: absolute as is, else under [`Self::data_root`].
    pub fn data_path(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.data_root().join(p)
        }
    }

    /// OpenVINO install dir to use: `openvino_dir` (resolved against the exe dir), else
    /// `<download_dir>/openvino` when `download_dir` is set and that exists. None = the exe
    /// dir's `openvino/`, then the system.
    pub fn openvino_dir_effective(&self) -> Option<PathBuf> {
        if let Some(d) = &self.openvino_dir {
            return Some(crate::resolve_path(d));
        }
        self.download_dir.as_ref()?;
        let d = self
            .data_root()
            .join(crate::backend::libs::BUNDLED_DIR_NAME);
        d.is_dir().then_some(d)
    }

    /// ONNX Runtime lookup options for this config (explicit dir and the install root).
    pub fn ort_options(&self) -> crate::backend::OrtOptions {
        crate::backend::OrtOptions {
            onnxruntime_dir: self.onnxruntime_dir.as_deref().map(crate::resolve_path),
            default_dir: self
                .download_dir
                .is_some()
                .then(|| self.data_root().join(crate::backend::libs::ORT_DIR_NAME)),
            cuda_libs_dir: Some(
                self.data_root()
                    .join(crate::backend::libs::ORT_DIR_NAME)
                    .join(crate::backend::libs::CUDA_LIBS_DIR_NAME),
            ),
        }
    }

    /// The hardware profile of this machine under `hardware_profile`.
    pub fn profile(&self) -> crate::profile::Profile {
        crate::profile::Profile::current(self.hardware_profile)
    }

    /// Explicit concurrency settings of `m` (its `inflight`, else the global one; 0 = profile).
    pub fn concurrency_overrides(&self, m: &ModelConfig) -> crate::profile::Overrides {
        crate::profile::Overrides {
            inflight: m.inflight.filter(|&n| n > 0).unwrap_or(self.inflight),
            ort_sessions: self.ort_sessions,
        }
    }

    /// Cache directory resolved against the exe dir, or None when disabled.
    pub fn cache_dir_path(&self) -> Option<PathBuf> {
        if self.cache_dir.trim().is_empty() {
            None
        } else {
            Some(crate::resolve_path(Path::new(&self.cache_dir)))
        }
    }
}

/// Catalog model a config without models starts with ([`Config::seed_default_model`]).
pub const DEFAULT_MODEL_NAME: &str = "IPcam-general";

/// Accepted range of `intra_threads` and `preprocess_threads`.
pub const MAX_THREADS: usize = 256;
/// Accepted range of `inflight` (global and per model).
pub const MAX_INFLIGHT_SETTING: usize = 64;
/// Accepted range of `ort_sessions`.
pub const MAX_ORT_SESSIONS: usize = 16;
/// Accepted range of `gpu_index`.
pub const MAX_GPU_INDEX: u32 = 64;
/// Accepted range of `request_timeout_secs`.
pub const MAX_REQUEST_TIMEOUT_SECS: u64 = 3600;
/// Accepted range of `max_image_pixels`.
pub const MIN_IMAGE_PIXELS: u64 = 1_000;
pub const MAX_IMAGE_PIXELS: u64 = 1_000_000_000;
/// Default of `save_image_max_age_days`.
pub const DEFAULT_SAVE_IMAGE_MAX_AGE_DAYS: u32 = 30;
/// Accepted range of `save_image_max_age_days` (10 years; 0 = unlimited).
pub const MAX_SAVE_IMAGE_AGE_DAYS: u32 = 3650;
/// Default of `save_image_max_mb`.
pub const DEFAULT_SAVE_IMAGE_MAX_MB: u64 = 2048;
/// Accepted range of `save_image_max_mb` (about 1 TB; 0 = unlimited).
pub const MAX_SAVE_IMAGE_MB: u64 = 1_000_000;
/// Accepted range of `benchmark.warmup` and `benchmark.repeat_per_image`.
pub const MAX_BENCH_ITERATIONS: usize = 1000;

/// `field` must lie in `lo..=hi`; the error names the key and the range.
pub fn check_range<T: PartialOrd + std::fmt::Display>(
    field: &str,
    v: T,
    lo: T,
    hi: T,
) -> Result<()> {
    if v < lo || v > hi {
        anyhow::bail!("{field}: must be between {lo} and {hi}, got {v}; set a value in that range");
    }
    Ok(())
}

/// The one rule for confidence thresholds and `nms_iou` (global and per model): a finite
/// number in `0..=1`. 0 is allowed (report every box).
pub fn check_threshold(field: &str, t: f32) -> Result<()> {
    if !(t.is_finite() && (0.0..=1.0).contains(&t)) {
        anyhow::bail!(
            "{field}: must be between 0 and 1, got {t}; use a fraction such as 0.5 (not a \
             percentage)"
        );
    }
    Ok(())
}

/// Every model entry on its own (an empty list is fine here): a non-empty `path`, distinct
/// names, valid per-model device specs, thresholds per [`check_threshold`] and `inflight` in
/// range.
pub fn validate_model_entries(models: &[ModelConfig]) -> Result<()> {
    use anyhow::bail;
    let mut seen = std::collections::HashSet::new();
    for m in models {
        if m.path.as_os_str().is_empty() {
            bail!(
                "models: an entry has no `path`; set it to the model file (.onnx or .xml) or \
                 remove the entry"
            );
        }
        let key = crate::registry::normalize_name(&m.effective_name());
        if !seen.insert(key.clone()) {
            bail!("models: duplicate model name '{key}'; set a distinct `name` for each entry");
        }
        if let Some(d) = &m.device {
            crate::backend::spec::parse(d).map_err(|e| {
                anyhow::anyhow!(
                    "models: '{key}' device: {e}; run `blue-onyx-prism list-devices` for the \
                     specs this machine accepts, or remove the key to use the global device"
                )
            })?;
        }
        if let Some(t) = m.confidence_threshold {
            check_threshold(&format!("models: '{key}' confidence_threshold"), t)?;
        }
        if let Some(n) = m.inflight {
            check_range(
                &format!("models: '{key}' inflight"),
                n,
                0,
                MAX_INFLIGHT_SETTING,
            )?;
        }
        validate_filter_rules(
            &format!("models: '{key}' "),
            &FilterFields {
                night_confidence_threshold: m.night_confidence_threshold,
                label_thresholds: m.label_thresholds.as_ref(),
                night_label_thresholds: m.night_label_thresholds.as_ref(),
                person_aspect: m.person_aspect,
                min_box_area_frac: m.min_box_area_frac,
                suppress_overlaps: m.suppress_overlaps.as_deref(),
                static_suppression: m.static_suppression.as_ref(),
            },
        )?;
    }
    validate_night_models(models)?;
    for m in models {
        if let Some(v) = &m.verify {
            let key = crate::registry::normalize_name(&m.effective_name());
            let field = format!("models: '{key}' verify");
            v.validate(&field)?;
            if v.enabled && crate::registry::normalize_name(&v.model) == key {
                anyhow::bail!(
                    "{field}.model: names the model itself; name another configured model (one \
                     that makes different mistakes, e.g. another family) or remove `verify`"
                );
            }
            check_verify_model(&field, v, models)?;
        }
    }
    Ok(())
}

/// The global `verify`: ranges and a configured `model`.
pub fn validate_verify(
    v: Option<&crate::verify::VerifyConfig>,
    models: &[ModelConfig],
) -> Result<()> {
    let Some(v) = v else {
        return Ok(());
    };
    v.validate("verify")?;
    check_verify_model("verify", v, models)
}

/// An enabled `verify` names a configured model (enabled or not: a disabled or unloaded
/// verifier leaves the candidates unverified).
fn check_verify_model(
    field: &str,
    v: &crate::verify::VerifyConfig,
    models: &[ModelConfig],
) -> Result<()> {
    use crate::registry::normalize_name;
    if !v.enabled {
        return Ok(());
    }
    let key = normalize_name(&v.model);
    if !models
        .iter()
        .any(|m| normalize_name(&m.effective_name()) == key)
    {
        anyhow::bail!(
            "{field}.model: '{}' is not a configured model; add it to `models` first, or fix the \
             name (see the Models card)",
            v.model
        );
    }
    Ok(())
}

/// `night_model` of every entry names another entry (any, enabled or not: a disabled night
/// model only makes the switch fall back) that has no `night_model` itself (one hop: no chains
/// or cycles).
pub fn validate_night_models(models: &[ModelConfig]) -> Result<()> {
    use crate::registry::normalize_name;
    use anyhow::bail;
    let night_of = |key: &str| {
        models
            .iter()
            .find(|m| normalize_name(&m.effective_name()) == key)
    };
    for m in models {
        let Some(night) = m.night_model.as_deref() else {
            continue;
        };
        let key = normalize_name(&m.effective_name());
        let field = format!("models: '{key}' night_model");
        if night.trim().is_empty() {
            bail!("{field}: is empty; name another model or remove the key");
        }
        let nkey = normalize_name(night);
        if nkey == key {
            bail!(
                "{field}: names the model itself; name another configured model (e.g. one \
                 trained on IR frames) or remove the key"
            );
        }
        let Some(target) = night_of(&nkey) else {
            bail!(
                "{field}: '{night}' is not a configured model; add it to `models` first, or fix \
                 the name (see the Models card)"
            );
        };
        if target.night_model.is_some() {
            bail!(
                "{field}: '{night}' has a night_model of its own; night models do not chain, so \
                 remove night_model from '{night}' or point '{key}' at a model without one"
            );
        }
    }
    Ok(())
}

/// Largest accepted `person_aspect` bound (width / height).
pub const MAX_PERSON_ASPECT: f32 = 20.0;

/// The night / IR rule fields of the config or of one model entry (see
/// [`validate_filter_rules`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct FilterFields<'a> {
    pub night_confidence_threshold: Option<f32>,
    pub label_thresholds: Option<&'a BTreeMap<String, f32>>,
    pub night_label_thresholds: Option<&'a BTreeMap<String, f32>>,
    pub person_aspect: Option<[f32; 2]>,
    pub min_box_area_frac: Option<f32>,
    pub suppress_overlaps: Option<&'a [OverlapRule]>,
    pub static_suppression: Option<&'a StaticSuppression>,
}

/// Range checks of the night / IR rules; errors start with `prefix` (empty for the globals,
/// `models: 'x' ` for a model) and name the key.
pub fn validate_filter_rules(prefix: &str, f: &FilterFields<'_>) -> Result<()> {
    use anyhow::bail;
    if let Some(t) = f.night_confidence_threshold {
        check_threshold(&format!("{prefix}night_confidence_threshold"), t)?;
    }
    for (field, map) in [
        ("label_thresholds", f.label_thresholds),
        ("night_label_thresholds", f.night_label_thresholds),
    ] {
        for (label, t) in map.into_iter().flatten() {
            if label.trim().is_empty() {
                bail!("{prefix}{field}: has an empty label; use a class name such as \"person\"");
            }
            check_threshold(&format!("{prefix}{field}.{label}"), *t)?;
        }
    }
    if let Some([lo, hi]) = f.person_aspect {
        let field = format!("{prefix}person_aspect");
        if !(lo.is_finite() && hi.is_finite() && lo >= 0.0 && hi <= MAX_PERSON_ASPECT && lo < hi) {
            bail!(
                "{field}: must be [min, max] width/height with 0 <= min < max <= \
                 {MAX_PERSON_ASPECT}, got [{lo}, {hi}]; a standing person is about [0.2, 0.9]"
            );
        }
    }
    if let Some(a) = f.min_box_area_frac {
        check_threshold(&format!("{prefix}min_box_area_frac"), a)?;
    }
    for (i, r) in f.suppress_overlaps.into_iter().flatten().enumerate() {
        let field = format!("{prefix}suppress_overlaps[{i}]");
        if r.drop.trim().is_empty() || r.if_overlaps.trim().is_empty() {
            bail!(
                "{field}: `drop` and `if_overlaps` must both name a label, e.g. \
                 {{\"drop\": \"person\", \"if_overlaps\": \"potted plant\"}}"
            );
        }
        check_threshold(&format!("{field}.iou"), r.iou)?;
        if r.iou <= 0.0 {
            bail!("{field}.iou: must be above 0 (0 would match boxes that do not touch), got 0");
        }
    }
    if let Some(s) = f.static_suppression {
        s.validate(&format!("{prefix}static_suppression"))?;
    }
    Ok(())
}

/// The models list on its own: at least one entry, and [`validate_model_entries`].
pub fn validate_models(models: &[ModelConfig]) -> Result<()> {
    if models.is_empty() {
        anyhow::bail!("{NO_MODELS}");
    }
    validate_model_entries(models)
}

impl Config {
    /// Range-check every field (see the README's Configuration table). Each error names the key
    /// and the accepted range, e.g. `confidence_threshold: must be between 0 and 1, got 5`.
    /// An empty models list is fine here (first start); [`validate_config`] adds the checks a
    /// config the server runs with must pass.
    pub fn validate(&self) -> Result<()> {
        use anyhow::bail;
        check_range("port", self.port, 1, u16::MAX)?;
        check_range(
            "request_timeout_secs",
            self.request_timeout_secs,
            1,
            MAX_REQUEST_TIMEOUT_SECS,
        )?;
        check_range(
            "worker_queue_size",
            self.worker_queue_size,
            0,
            crate::worker::MAX_QUEUE_SIZE,
        )?;
        check_threshold("confidence_threshold", self.confidence_threshold)?;
        check_threshold("nms_iou", self.nms_iou)?;
        check_range("intra_threads", self.intra_threads, 0, MAX_THREADS)?;
        check_range(
            "preprocess_threads",
            self.preprocess_threads,
            0,
            MAX_THREADS,
        )?;
        check_range("inflight", self.inflight, 0, MAX_INFLIGHT_SETTING)?;
        check_range("ort_sessions", self.ort_sessions, 0, MAX_ORT_SESSIONS)?;
        check_range("gpu_index", self.gpu_index, 0, MAX_GPU_INDEX)?;
        check_range(
            "max_image_pixels",
            self.max_image_pixels,
            MIN_IMAGE_PIXELS,
            MAX_IMAGE_PIXELS,
        )?;
        check_range(
            "save_image_max_age_days",
            self.save_image_max_age_days,
            0,
            MAX_SAVE_IMAGE_AGE_DAYS,
        )?;
        check_range(
            "save_image_max_mb",
            self.save_image_max_mb,
            0,
            MAX_SAVE_IMAGE_MB,
        )?;
        if self.models_dir.as_os_str().is_empty() {
            bail!("models_dir: must not be empty; set a folder such as \"models\"");
        }
        crate::backend::spec::parse(&self.device).map_err(|e| {
            anyhow::anyhow!(
                "device: {e}; run `blue-onyx-prism list-devices` for the specs this machine \
                 accepts, or use \"auto\""
            )
        })?;
        validate_filter_rules(
            "",
            &FilterFields {
                night_confidence_threshold: self.night_confidence_threshold,
                label_thresholds: Some(&self.label_thresholds),
                night_label_thresholds: Some(&self.night_label_thresholds),
                person_aspect: self.person_aspect,
                min_box_area_frac: self.min_box_area_frac,
                suppress_overlaps: Some(&self.suppress_overlaps),
                static_suppression: Some(&self.static_suppression),
            },
        )?;
        validate_model_entries(&self.models)?;
        validate_verify(self.verify.as_ref(), &self.models)?;
        self.collect.validate(&self.data_root())?;
        let b = &self.benchmark;
        check_range("benchmark.warmup", b.warmup, 0, MAX_BENCH_ITERATIONS)?;
        check_range(
            "benchmark.repeat_per_image",
            b.repeat_per_image,
            0,
            MAX_BENCH_ITERATIONS,
        )?;
        for (field, w) in [
            ("benchmark.weights.accuracy", b.weights.accuracy),
            ("benchmark.weights.speed", b.weights.speed),
        ] {
            if !(w.is_finite() && (0.0..=1.0).contains(&w)) {
                bail!("{field}: must be between 0 and 1, got {w}");
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Path settings (SV-03) and the file browser's path syntax (SV-10)

/// Longest accepted path setting, in bytes.
pub const MAX_PATH_LEN: usize = 4096;
/// Longest accepted path component, in bytes.
pub const MAX_PATH_COMPONENT_LEN: usize = 255;

/// The settings that name a directory and go through [`validate_dir_setting`].
pub const DIR_SETTINGS: &[&str] = &[
    "save_image_path",
    "log_path",
    "cache_dir",
    "download_dir",
    "models_dir",
];

/// Directory settings that must not be empty.
const REQUIRED_DIR_SETTINGS: &[&str] = &["models_dir"];

/// Directories the service must never write to or list from, as the target or an ancestor
/// (matched on both platforms' spellings, so a Windows path typed on Linux is refused too).
const SYSTEM_DIRS: &[&str] = &["/etc", "/bin", "/sbin", "/usr", "/system", "/library"];
/// Windows system directories, matched on any drive letter: `<x>:/windows`,
/// `<x>:/program files*`.
const WINDOWS_SYSTEM_DIRS: &[&str] = &["windows", "program files"];

/// `p` with `\` as `/`, lowercased, without a trailing separator (for prefix comparisons).
fn normalized(p: &str) -> String {
    let s = p.replace('\\', "/").to_ascii_lowercase();
    let t = s.trim_end_matches('/');
    if t.is_empty() { s } else { t.to_string() }
}

/// `C:` / `c:` at the start of `s` (a Windows drive).
fn drive_prefix(s: &str) -> bool {
    matches!(s.as_bytes(), [d, b':', ..] if d.is_ascii_alphabetic())
}

/// Absolute on either platform: `/x`, `\x`, `C:\x`, `C:/x`.
pub fn is_absolute_path_text(s: &str) -> bool {
    if s.starts_with('/') || s.starts_with('\\') {
        return true;
    }
    drive_prefix(s)
        && s.as_bytes()
            .get(2)
            .is_some_and(|c| *c == b'\\' || *c == b'/')
}

/// Syntax checks shared by the path settings and the file browser: no NUL bytes, at most
/// [`MAX_PATH_LEN`] bytes and [`MAX_PATH_COMPONENT_LEN`] per component, no `..` component, no
/// Windows verbatim (`\\?\`) or UNC (`\\server\share`) prefix and no drive-relative path
/// (`C:foo`). `key` names the setting in the error.
pub fn check_path_syntax(key: &str, raw: &str) -> Result<()> {
    use anyhow::bail;
    if raw.contains('\0') {
        bail!("{key}: the path contains a NUL byte");
    }
    if raw.len() > MAX_PATH_LEN {
        bail!("{key}: the path is longer than {MAX_PATH_LEN} bytes");
    }
    if raw.starts_with(r"\\?\") || raw.starts_with(r"\??\") || raw.starts_with("//?/") {
        bail!("{key}: verbatim paths (\\\\?\\) are not accepted");
    }
    if raw.starts_with(r"\\") || raw.starts_with("//") {
        bail!("{key}: UNC paths (\\\\server\\share) are not accepted");
    }
    if drive_prefix(raw) && !is_absolute_path_text(raw) {
        bail!(
            "{key}: drive-relative paths ({}) are not accepted; give the full path",
            &raw[..2]
        );
    }
    for part in raw.split(['/', '\\']) {
        if part == ".." {
            bail!("{key}: '..' is not accepted in a path");
        }
        if part.len() > MAX_PATH_COMPONENT_LEN {
            bail!("{key}: a path component is longer than {MAX_PATH_COMPONENT_LEN} bytes");
        }
    }
    Ok(())
}

/// Whether `p` (absolute, either platform's spelling) is one of the system directories or
/// inside one (see `SYSTEM_DIRS`, `WINDOWS_SYSTEM_DIRS`).
pub fn is_system_dir(p: &Path) -> bool {
    let n = normalized(&p.to_string_lossy());
    let under = |dir: &str| n == dir || n.starts_with(&format!("{dir}/"));
    if SYSTEM_DIRS.iter().any(|d| under(d)) {
        return true;
    }
    // `c:/windows`, `c:/program files`, `c:/program files (x86)`.
    let Some(rest) = n
        .get(3..)
        .filter(|_| drive_prefix(&n) && n.as_bytes().get(2) == Some(&b'/'))
    else {
        return false;
    };
    WINDOWS_SYSTEM_DIRS.iter().any(|d| {
        rest == *d
            || rest.starts_with(&format!("{d}/"))
            || (*d == "program files" && rest.starts_with(d))
    })
}

/// Validate a directory setting typed into the Config page (`key` is one of
/// [`DIR_SETTINGS`]): `Ok(None)` for an empty value (an error for a required one),
/// [`check_path_syntax`], relative paths resolved under `data_root`, and a path that is not
/// under the data root must not be a system directory or inside one ([`is_system_dir`]).
/// Returns the path to store (absolute).
pub fn validate_dir_setting(key: &str, raw: &str, data_root: &Path) -> Result<Option<PathBuf>> {
    use anyhow::bail;
    let raw = raw.trim();
    let raw = ['"', '\'']
        .iter()
        .find(|q| raw.len() >= 2 && raw.starts_with(**q) && raw.ends_with(**q))
        .map_or(raw, |_| raw.get(1..raw.len() - 1).unwrap_or(raw).trim());
    if raw.is_empty() {
        if REQUIRED_DIR_SETTINGS.contains(&key) {
            bail!("{key}: must not be empty");
        }
        return Ok(None);
    }
    check_path_syntax(key, raw)?;
    let resolved = if is_absolute_path_text(raw) {
        PathBuf::from(raw)
    } else {
        data_root.join(raw)
    };
    let under_root = !data_root.as_os_str().is_empty()
        && normalized(&resolved.to_string_lossy())
            .starts_with(&format!("{}/", normalized(&data_root.to_string_lossy())));
    if !under_root && is_system_dir(&resolved) {
        bail!(
            "{key}: {} is a system directory; choose a folder of your own (for example under {})",
            resolved.display(),
            data_root.display()
        );
    }
    Ok(Some(resolved))
}

/// Error of an empty models list.
const NO_MODELS: &str = "models: at least one model is required; add one on the Config page \
     (Add a model by path), with `blue-onyx-prism add-model --path <file>`, or fetch the default \
     with `blue-onyx-prism download-models --name IPcam-general --add-to-config`";

/// Checks of a config the server runs with: [`Config::validate`], at least one model, at least
/// one enabled model and a `default_model` that names a configured model.
pub fn validate_config(c: &Config) -> Result<()> {
    use anyhow::bail;
    c.validate()?;
    if c.models.is_empty() {
        bail!("{NO_MODELS}");
    }
    if !c.models.iter().any(|m| m.enabled) {
        bail!(
            "models: at least one model must be enabled; tick Load for a model on the Models \
             card (or set \"enabled\": true)"
        );
    }
    if let Some(name) = &c.default_model
        && c.default_model_index().is_none()
    {
        bail!(
            "default_model: '{name}' is not one of the configured models (renamed or \
             removed?); pick one of the configured models or clear default_model"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn log_timezone_parses_every_form() {
        assert_eq!(" UTC ".parse::<LogTimezone>().unwrap(), LogTimezone::Utc);
        assert_eq!("Local".parse::<LogTimezone>().unwrap(), LogTimezone::Local);
        let la: LogTimezone = "America/Los_Angeles".parse().unwrap();
        assert!(matches!(la, LogTimezone::Named(_)));
        assert_eq!(la.to_string(), "America/Los_Angeles");
        for (s, shown) in [
            ("+08:00", "+08:00"),
            ("-05:30", "-05:30"),
            ("+00:00", "+00:00"),
        ] {
            assert_eq!(s.parse::<LogTimezone>().unwrap().to_string(), shown);
        }
        for bad in [
            "",
            "mars",
            "+8",
            "+25:00",
            "+08:60",
            "08:00",
            "+0800",
            "Nowhere/City",
        ] {
            let e = bad.parse::<LogTimezone>().unwrap_err().to_string();
            assert!(
                e.contains("log_timezone") && e.contains("IANA"),
                "{bad}: {e}"
            );
        }
    }

    #[test]
    fn log_timezone_formats_a_fixed_instant() {
        use chrono::TimeZone;
        let t = chrono::Utc
            .with_ymd_and_hms(2026, 10, 16, 0, 54, 59)
            .unwrap()
            + chrono::Duration::milliseconds(580);
        let f = |s: &str| s.parse::<LogTimezone>().unwrap().format(t);
        assert_eq!(f("utc"), "2026-10-16T00:54:59.580Z");
        assert_eq!(f("+08:00"), "2026-10-16T08:54:59.580+08:00");
        assert_eq!(f("-05:30"), "2026-10-15T19:24:59.580-05:30");
        // DST: PDT (-07:00) in October, PST (-08:00) in January.
        assert_eq!(f("America/Los_Angeles"), "2026-10-15T17:54:59.580-07:00");
        let winter = chrono::Utc.with_ymd_and_hms(2026, 1, 16, 0, 0, 0).unwrap();
        assert_eq!(
            "America/Los_Angeles"
                .parse::<LogTimezone>()
                .unwrap()
                .format(winter),
            "2026-01-15T16:00:00.000-08:00"
        );
        assert!(chrono::DateTime::parse_from_rfc3339(&f("local")).is_ok());
    }

    #[test]
    fn log_timezone_config_round_trip() {
        let c: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(c.log_timezone, LogTimezone::Utc);
        let c: Config = serde_json::from_str(r#"{"log_timezone":"Europe/Berlin"}"#).unwrap();
        let text = serde_json::to_string(&c).unwrap();
        assert!(text.contains(r#""log_timezone":"Europe/Berlin""#), "{text}");
        let back: Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back.log_timezone, c.log_timezone);
        let err = serde_json::from_str::<Config>(r#"{"log_timezone":"mars"}"#)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("log_timezone") && err.contains("mars"),
            "{err}"
        );
    }
    #[test]
    fn dir_setting_accepts_own_folders() {
        use super::validate_dir_setting;
        use std::path::{Path, PathBuf};
        let root = Path::new("/srv/prism");
        assert_eq!(validate_dir_setting("log_path", "", root).unwrap(), None);
        assert_eq!(validate_dir_setting("cache_dir", "  ", root).unwrap(), None);
        assert_eq!(
            validate_dir_setting("log_path", "logs", root).unwrap(),
            Some(root.join("logs"))
        );
        assert_eq!(
            validate_dir_setting("save_image_path", "\"/data/shots\"", root).unwrap(),
            Some(PathBuf::from("/data/shots"))
        );
        // Under the data root anything goes, even when the data root is under /usr.
        let usr = Path::new("/usr/local/prism");
        assert_eq!(
            validate_dir_setting("cache_dir", "cache", usr).unwrap(),
            Some(usr.join("cache"))
        );
        assert!(validate_dir_setting("cache_dir", "/usr/local/prism/x", usr).is_ok());
        assert!(validate_dir_setting("models_dir", "C:\\prism\\models", root).is_ok());
        assert!(validate_dir_setting("models_dir", "/home/me/Program Files", root).is_ok());
        assert!(validate_dir_setting("models_dir", "/etcetera", root).is_ok());
        let err = validate_dir_setting("models_dir", "", root).unwrap_err();
        assert!(err.to_string().contains("models_dir: must not be empty"));
    }

    #[test]
    fn dir_setting_rejects_nul() {
        let e = super::validate_dir_setting("log_path", "a\0b", std::path::Path::new("/r"))
            .unwrap_err();
        assert!(e.to_string().contains("NUL"), "{e}");
    }

    #[test]
    fn dir_setting_rejects_long_component() {
        let long = "x".repeat(256);
        let e =
            super::validate_dir_setting("log_path", &long, std::path::Path::new("/r")).unwrap_err();
        assert!(e.to_string().contains("component"), "{e}");
        assert!(
            super::validate_dir_setting("log_path", &"y".repeat(255), std::path::Path::new("/r"))
                .is_ok()
        );
    }

    #[test]
    fn dir_setting_rejects_long_path() {
        let long = "ab/".repeat(1400);
        let e =
            super::validate_dir_setting("log_path", &long, std::path::Path::new("/r")).unwrap_err();
        assert!(e.to_string().contains("longer than 4096"), "{e}");
    }

    #[test]
    fn dir_setting_rejects_verbatim_prefix() {
        let r = std::path::Path::new("/r");
        for p in [r"\\?\C:\x", r"\??\C:\x", "//?/C:/x"] {
            let e = super::validate_dir_setting("log_path", p, r).unwrap_err();
            assert!(e.to_string().contains("verbatim"), "{p}: {e}");
        }
    }

    #[test]
    fn dir_setting_rejects_unc_prefix() {
        let r = std::path::Path::new("/r");
        for p in [r"\\server\share\x", "//server/share"] {
            let e = super::validate_dir_setting("log_path", p, r).unwrap_err();
            assert!(e.to_string().contains("UNC"), "{p}: {e}");
        }
    }

    #[test]
    fn dir_setting_rejects_drive_relative() {
        let r = std::path::Path::new("/r");
        for p in ["C:foo", "c:", "D:..\\x"] {
            let e = super::validate_dir_setting("log_path", p, r).unwrap_err();
            assert!(e.to_string().contains("drive-relative"), "{p}: {e}");
        }
    }

    #[test]
    fn dir_setting_rejects_dot_dot() {
        let r = std::path::Path::new("/r");
        for p in ["../x", "a/../b", "/data/../etc", r"C:\x\..\y", ".."] {
            let e = super::validate_dir_setting("log_path", p, r).unwrap_err();
            assert!(e.to_string().contains("'..'"), "{p}: {e}");
        }
        // A name that merely contains dots is fine.
        assert!(super::validate_dir_setting("log_path", "a/..b/...", r).is_ok());
    }

    #[test]
    fn dir_setting_rejects_system_dirs() {
        let r = std::path::Path::new("/srv/prism");
        for p in [
            "/etc",
            "/etc/",
            "/ETC/cron.d",
            "/bin",
            "/sbin/x",
            "/usr",
            "/usr/share/x",
            "/System/Library",
            "/Library/LaunchDaemons",
            r"C:\Windows",
            r"c:\windows\system32\drivers\etc",
            "C:/Windows/Temp",
            r"C:\Program Files",
            r"C:\Program Files (x86)\x",
            r"D:\Program Files\y",
        ] {
            let e = super::validate_dir_setting("save_image_path", p, r).unwrap_err();
            assert!(e.to_string().contains("system directory"), "{p}: {e}");
        }
        assert!(!super::is_system_dir(std::path::Path::new(
            "/var/lib/prism"
        )));
        assert!(!super::is_system_dir(std::path::Path::new(
            r"C:\Users\me\Program Files"
        )));
        assert!(!super::is_system_dir(std::path::Path::new(r"C:\Windowsx")));
    }

    #[test]
    fn threshold_objective_parsing_and_defaults() {
        use crate::benchmark::threshold::Objective;
        // Old files (no key, or a benchmark section without it) get f1.
        let c: super::Config = serde_json::from_str(r#"{"benchmark":{"warmup":2}}"#).unwrap();
        assert_eq!(c.benchmark.threshold_objective, Objective::F1);
        assert_eq!(c.benchmark.warmup, 2);
        let c: super::Config =
            serde_json::from_str(r#"{"benchmark":{"threshold_objective":"precision:0.9"}}"#)
                .unwrap();
        assert_eq!(c.benchmark.threshold_objective, Objective::Precision(0.9));
        let text = serde_json::to_string(&c).unwrap();
        assert!(
            text.contains(r#""threshold_objective":"precision:0.9""#),
            "{text}"
        );
        let back: super::Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        // A bad objective is an error, not a silent default.
        assert!(
            serde_json::from_str::<super::Config>(r#"{"benchmark":{"threshold_objective":"f3"}}"#)
                .is_err()
        );
    }

    #[test]
    fn accuracy_metric_parsing_and_defaults() {
        use crate::benchmark::grade::AccuracyMetric;
        use crate::benchmark::threshold::Objective;
        // Old files get ap50.
        let c: super::Config = serde_json::from_str(r#"{"benchmark":{"warmup":2}}"#).unwrap();
        assert_eq!(c.benchmark.accuracy_metric, AccuracyMetric::Ap50);
        let c: super::Config = serde_json::from_str(r#"{"port":1}"#).unwrap();
        assert_eq!(c.benchmark.accuracy_metric, AccuracyMetric::Ap50);
        let c: super::Config = serde_json::from_str(
            r#"{"benchmark":{"accuracy_metric":"roc_auc","threshold_objective":"youden"}}"#,
        )
        .unwrap();
        assert_eq!(c.benchmark.accuracy_metric, AccuracyMetric::RocAuc);
        assert_eq!(c.benchmark.threshold_objective, Objective::Youden);
        let text = serde_json::to_string(&c).unwrap();
        assert!(text.contains(r#""accuracy_metric":"roc_auc""#), "{text}");
        assert!(text.contains(r#""threshold_objective":"youden""#), "{text}");
        let back: super::Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        let c: super::Config =
            serde_json::from_str(r#"{"benchmark":{"threshold_objective":"fpr:0.05"}}"#).unwrap();
        assert_eq!(c.benchmark.threshold_objective, Objective::Fpr(0.05));
        assert!(
            serde_json::from_str::<super::Config>(r#"{"benchmark":{"accuracy_metric":"map"}}"#)
                .is_err()
        );
    }

    #[test]
    fn benchmark_section_defaults_and_round_trip() {
        // Old files without the key get the defaults.
        let c: super::Config = serde_json::from_str(r#"{"port":1234}"#).unwrap();
        assert_eq!(c.benchmark, super::BenchmarkConfig::default());
        let ids: Vec<String> = c.benchmark.datasets.iter().map(|d| d.label()).collect();
        assert_eq!(ids, ["coco-cctv", "bmd45-cctv", "exdark-night"]);
        assert_eq!((c.benchmark.warmup, c.benchmark.repeat_per_image), (3, 1));
        assert!(c.benchmark.auto_download_datasets);
        assert!((c.benchmark.weights.accuracy - 0.6).abs() < 1e-12);
        assert!(c.benchmark.low_priority);
        let c: super::Config =
            serde_json::from_str(r#"{"benchmark":{"low_priority":false}}"#).unwrap();
        assert!(!c.benchmark.low_priority);
        assert_eq!(c.benchmark.warmup, 3);
        let text = serde_json::to_string(&c).unwrap();
        assert!(text.contains(r#""low_priority":false"#), "{text}");
        // Partial sections keep the other defaults; all dataset spellings parse.
        let c: super::Config = serde_json::from_str(
            r#"{"benchmark":{"datasets":["coco-cctv","dir:/srv/alerts",
                {"dir":"/srv/x","gt":"/srv/x/gt.json","name":"porch"}],
                "max_images_per_dataset":25,"weights":{"accuracy":1,"speed":1}}}"#,
        )
        .unwrap();
        let b = &c.benchmark;
        assert_eq!(b.max_images_per_dataset, 25);
        assert_eq!(b.warmup, 3);
        assert_eq!(b.datasets[1].dir().unwrap().0, PathBuf::from("/srv/alerts"));
        assert_eq!(b.datasets[2].label(), "porch");
        assert_eq!(
            b.datasets[2].dir().unwrap().1,
            Some(PathBuf::from("/srv/x/gt.json"))
        );
        assert!(b.datasets[0].dir().is_none());
        let text = serde_json::to_string(&c).unwrap();
        let back: super::Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn ort_lookup_searches_download_dir() {
        // The status page and the loader share `ort_options().lookup()`; an install under
        // `download_dir` (not next to the exe) must be found.
        if std::env::var_os(crate::backend::libs::ENV_ORT_DYLIB_PATH).is_some() {
            return;
        }
        let root = std::env::temp_dir().join(format!("bop-ortdl-{}", uuid::Uuid::new_v4()));
        let flavor = root.join(crate::backend::libs::ORT_DIR_NAME).join("cpu");
        std::fs::create_dir_all(&flavor).unwrap();
        let lib = flavor.join(crate::backend::libs::ort_library_file_name());
        std::fs::write(&lib, b"").unwrap();
        let c = super::Config {
            download_dir: Some(root.clone()),
            ..Default::default()
        };
        assert_eq!(
            c.ort_options().lookup().library.as_deref(),
            Some(lib.as_path())
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn legacy_config_is_renamed_once() {
        let dir = std::env::temp_dir().join(format!("bop-legacy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // No files: the current name is returned and nothing is created.
        let p = super::migrate_legacy(&dir, super::CONFIG_FILE, super::LEGACY_CONFIG_FILE);
        assert_eq!(p, dir.join(super::CONFIG_FILE));
        assert!(!p.exists());
        // Only the legacy file: it is renamed.
        std::fs::write(dir.join(super::LEGACY_CONFIG_FILE), "{}").unwrap();
        let p = super::migrate_legacy(&dir, super::CONFIG_FILE, super::LEGACY_CONFIG_FILE);
        assert!(p.is_file() && !dir.join(super::LEGACY_CONFIG_FILE).exists());
        // Both present: the current one wins and the legacy file is left alone.
        std::fs::write(dir.join(super::LEGACY_CONFIG_FILE), "{\"port\":1}").unwrap();
        let p = super::migrate_legacy(&dir, super::CONFIG_FILE, super::LEGACY_CONFIG_FILE);
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "{}");
        assert!(dir.join(super::LEGACY_CONFIG_FILE).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    use super::*;

    #[test]
    fn roundtrip_and_defaults() {
        let mut c = Config::default();
        c.models.push(ModelConfig {
            path: "models/yolo26s.xml".into(),
            family: ModelFamilyKind::Yolo26,
            ..Default::default()
        });
        let s = serde_json::to_string(&c).unwrap();
        let back: Config = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
        assert_eq!(back.models[0].effective_name(), "yolo26s");
        assert_eq!(back.default_model_index(), Some(0));
        let named = Config {
            default_model: Some("YOLO26S.xml".into()),
            ..back.clone()
        };
        assert_eq!(named.default_model_index(), Some(0));
        let empty: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.port, crate::DEFAULT_PORT);
        // New configs default to `auto`; existing ones that say "GPU" keep OpenVINO GPU.
        assert_eq!(empty.device, "auto");
        assert_eq!(empty.onnxruntime_dir, None);
        let old: Config = serde_json::from_str(r#"{"device":"GPU"}"#).unwrap();
        assert_eq!(
            old.device_spec_for(&ModelConfig::default())
                .unwrap()
                .to_string(),
            "openvino:gpu"
        );
        let with_dir: Config =
            serde_json::from_str(r#"{"onnxruntime_dir":"rt/onnxruntime"}"#).unwrap();
        assert_eq!(
            with_dir.onnxruntime_dir,
            Some(PathBuf::from("rt/onnxruntime"))
        );
    }

    #[test]
    fn unnamed_models_are_named_on_load_and_save() {
        let dir = std::env::temp_dir().join(format!("bo-cfg-names-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.json");
        std::fs::write(
            &path,
            r#"{"models":[{"name":null,"path":"models/IPcam-general.onnx"},
                {"name":"custom","path":"models/x.onnx"},{"name":" ","path":"m/y.xml"}]}"#,
        )
        .unwrap();
        let c = Config::load(&path).unwrap();
        let names: Vec<_> = c.models.iter().map(|m| m.name.as_deref()).collect();
        assert_eq!(names, [Some("IPcam-general"), Some("custom"), Some("y")]);

        // Write-back names entries that were added without a name.
        let mut c = c;
        c.models.push(ModelConfig {
            path: "models/dfine-s.onnx".into(),
            ..Default::default()
        });
        c.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains(r#""name": "dfine-s""#), "{text}");
        assert!(!text.contains(r#""name": null"#), "{text}");
        // The in-memory config is left as it was.
        assert_eq!(c.models[3].name, None);
        assert_eq!(
            Config::load(&path).unwrap().models[3].effective_name(),
            "dfine-s"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn enabled_defaults_to_true_and_round_trips() {
        let m: ModelConfig = serde_json::from_str(r#"{"path":"models/a.onnx"}"#).unwrap();
        assert!(m.enabled);
        let c: Config =
            serde_json::from_str(r#"{"models":[{"path":"a.onnx"},{"path":"b.onnx"}]}"#).unwrap();
        assert!(c.models.iter().all(|m| m.enabled));

        let m: ModelConfig =
            serde_json::from_str(r#"{"path":"models/a.onnx","enabled":false}"#).unwrap();
        assert!(!m.enabled);
        let text = serde_json::to_string(&m).unwrap();
        assert!(text.contains(r#""enabled":false"#), "{text}");
        let back: ModelConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, m);
    }

    fn three_models() -> Config {
        Config {
            models: ["a", "b", "c"]
                .iter()
                .map(|n| ModelConfig {
                    path: format!("models/{n}.onnx").into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn effective_default_skips_disabled_models() {
        let mut c = three_models();
        assert_eq!(c.effective_default_index(), Some(0));
        c.models[0].enabled = false;
        assert_eq!(c.effective_default_index(), Some(1));
        c.default_model = Some("C".into());
        assert_eq!(c.effective_default_index(), Some(2));
        c.models[2].enabled = false;
        assert_eq!(c.effective_default_index(), Some(1));
        c.default_model = Some("nope".into());
        assert_eq!(c.effective_default_index(), Some(1));
        c.models[1].enabled = false;
        assert_eq!(c.effective_default_index(), None);
        assert_eq!(c.enabled_models().count(), 0);
    }

    #[test]
    fn add_model_if_absent_skips_duplicates() {
        let mut c = Config::default();
        let m = |name: &str, path: &str| ModelConfig {
            name: Some(name.into()),
            path: path.into(),
            ..Default::default()
        };
        assert!(c.add_model_if_absent(m("IPcam-general", "models/IPcam-general.onnx")));
        assert!(!c.add_model_if_absent(m("ipcam-general.onnx", "other/x.onnx")));
        assert!(!c.add_model_if_absent(m("different", "models/IPcam-general.onnx")));
        assert!(c.add_model_if_absent(ModelConfig {
            path: "models/delivery.onnx".into(),
            ..Default::default()
        }));
        assert_eq!(c.models.len(), 2);
    }

    #[test]
    fn device_resolution() {
        let mut c = Config::default();
        let m = ModelConfig::default();
        assert_eq!(c.device_for(&m), "auto");
        c.gpu_index = 1;
        assert_eq!(c.device_for(&m), "auto", "auto takes no index");
        c.device = "GPU".into();
        assert_eq!(c.device_for(&m), "GPU.1");
        c.force_cpu = true;
        assert_eq!(c.device_for(&m), "CPU");
        c.force_cpu = false;
        let m2 = ModelConfig {
            device: Some("CPU".into()),
            ..Default::default()
        };
        assert_eq!(c.device_for(&m2), "CPU");
    }

    #[test]
    fn device_spec_resolution() {
        use crate::backend::spec::parse;
        let spec = |c: &Config, m: &ModelConfig| c.device_spec_for(m).unwrap().to_string();
        let mut c = Config::default();
        let m = ModelConfig::default();
        assert_eq!(spec(&c, &m), "auto");
        c.device = "GPU".into();
        assert_eq!(spec(&c, &m), "openvino:gpu");
        c.gpu_index = 2;
        assert_eq!(c.device_for(&m), "GPU.2");
        assert_eq!(spec(&c, &m), "openvino:gpu.2");
        for (global, want) in [
            ("openvino:gpu", "openvino:gpu.2"),
            ("openvino:gpu.1", "openvino:gpu.1"),
            ("ort:cuda", "ort:cuda:2"),
            ("ort:directml", "ort:directml:2"),
            ("ORT:TensorRT", "ort:tensorrt:2"),
            ("ort:cuda:0", "ort:cuda:0"),
            ("ort:coreml", "ort:coreml"),
            ("CPU", "openvino:cpu"),
            ("auto", "auto"),
        ] {
            c.device = global.into();
            assert_eq!(spec(&c, &m), want, "{global}");
            assert_eq!(parse(&c.device_for(&m)).unwrap().to_string(), want);
        }
        // Per-model overrides are used verbatim (gpu_index only applies to the global device).
        let over = |d: &str| ModelConfig {
            device: Some(d.into()),
            ..Default::default()
        };
        assert_eq!(spec(&c, &over("GPU")), "openvino:gpu");
        assert_eq!(spec(&c, &over("ort:cpu")), "ort:cpu");
        // force_cpu wins over everything.
        c.force_cpu = true;
        assert_eq!(spec(&c, &over("ort:cuda")), "openvino:cpu");
        c.force_cpu = false;
        let err = c.device_spec_for(&over("G P U")).unwrap_err();
        assert!(format!("{err:#}").contains("device"), "{err:#}");
        c.device = "nonsense".into();
        assert!(c.device_spec_for(&m).is_err());
    }

    /// `validate` on a default config with `edit` applied: Ok, or the error text.
    fn check(edit: impl FnOnce(&mut Config)) -> std::result::Result<(), String> {
        let mut c = Config::default();
        edit(&mut c);
        c.validate().map_err(|e| format!("{e:#}"))
    }

    /// The error names `key` and the range.
    fn rejects(edit: impl FnOnce(&mut Config), key: &str, range: &str) {
        let err = check(edit).unwrap_err();
        assert!(err.contains(key), "{err}");
        assert!(err.contains(range), "{err}");
    }

    #[test]
    fn validate_accepts_defaults_and_an_empty_model_list() {
        assert_eq!(check(|_| {}), Ok(()));
        // `validate_config` (what the server runs with) still wants a model.
        assert!(validate_config(&Config::default()).is_err());
        assert!(validate_config(&three_models()).is_ok());
    }

    #[test]
    fn validate_port() {
        rejects(|c| c.port = 0, "port", "between 1 and 65535, got 0");
        assert!(check(|c| c.port = 65535).is_ok());
    }

    #[test]
    fn validate_request_timeout() {
        rejects(
            |c| c.request_timeout_secs = 0,
            "request_timeout_secs",
            "between 1 and 3600",
        );
        rejects(
            |c| c.request_timeout_secs = u64::MAX,
            "request_timeout_secs",
            "between 1 and 3600",
        );
        assert!(check(|c| c.request_timeout_secs = 3600).is_ok());
    }

    #[test]
    fn validate_worker_queue_size() {
        rejects(
            |c| c.worker_queue_size = 1_000_000_000,
            "worker_queue_size",
            "between 0 and 1024",
        );
        assert!(check(|c| c.worker_queue_size = 1024).is_ok());
        assert!(check(|c| c.worker_queue_size = 0).is_ok());
    }

    #[test]
    fn validate_night_model_references() {
        let entry = |name: &str, night: Option<&str>| ModelConfig {
            name: Some(name.into()),
            path: format!("{name}.onnx").into(),
            night_model: night.map(str::to_string),
            ..Default::default()
        };
        let err = |models: &[ModelConfig]| format!("{:#}", validate_models(models).unwrap_err());
        // A plain reference (case-insensitive, `.onnx` ignored); a disabled target is fine.
        let mut ok = vec![entry("day", Some("Night.onnx")), entry("night", None)];
        validate_models(&ok).unwrap();
        ok[1].enabled = false;
        validate_models(&ok).unwrap();
        // Two models may share one night model.
        validate_models(&[
            entry("a", Some("night")),
            entry("b", Some("night")),
            entry("night", None),
        ])
        .unwrap();
        let e = err(&[entry("day", Some("nope")), entry("night", None)]);
        assert!(e.contains("'nope' is not a configured model"), "{e}");
        let e = err(&[entry("day", Some("DAY"))]);
        assert!(e.contains("names the model itself"), "{e}");
        let e = err(&[entry("day", Some("  "))]);
        assert!(e.contains("night_model: is empty"), "{e}");
        // Chains and cycles.
        let e = err(&[
            entry("a", Some("b")),
            entry("b", Some("c")),
            entry("c", None),
        ]);
        assert!(
            e.contains("'a' night_model: 'b' has a night_model of its own"),
            "{e}"
        );
        let e = err(&[entry("a", Some("b")), entry("b", Some("a"))]);
        assert!(e.contains("do not chain"), "{e}");
        // Round trip: written only when set.
        let text = serde_json::to_string(&entry("day", None)).unwrap();
        assert!(!text.contains("night_model"), "{text}");
        let back: ModelConfig =
            serde_json::from_str(&serde_json::to_string(&ok[0]).unwrap()).unwrap();
        assert_eq!(back.night_model.as_deref(), Some("Night.onnx"));
    }

    #[test]
    fn validate_thresholds_global_and_per_model() {
        rejects(
            |c| c.confidence_threshold = 5.0,
            "confidence_threshold",
            "must be between 0 and 1, got 5",
        );
        rejects(
            |c| c.confidence_threshold = -1.0,
            "confidence_threshold",
            "0 and 1",
        );
        rejects(
            |c| c.confidence_threshold = f32::NAN,
            "confidence_threshold",
            "0 and 1",
        );
        rejects(|c| c.nms_iou = 2.0, "nms_iou", "0 and 1");
        assert!(check(|c| c.confidence_threshold = 0.0).is_ok());
        assert!(check(|c| c.nms_iou = 1.0).is_ok());
        let mut c = three_models();
        c.models[1].confidence_threshold = Some(1.5);
        let err = format!("{:#}", c.validate().unwrap_err());
        assert!(err.contains("'b' confidence_threshold"), "{err}");
        c.models[1].confidence_threshold = Some(0.0);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_save_image_retention() {
        let d = Config::default();
        assert_eq!(d.save_image_max_age_days, 30);
        assert_eq!(d.save_image_max_mb, 2048);
        rejects(
            |c| c.save_image_max_age_days = 3651,
            "save_image_max_age_days",
            "between 0 and 3650",
        );
        rejects(
            |c| c.save_image_max_mb = 1_000_001,
            "save_image_max_mb",
            "between 0 and 1000000",
        );
        assert!(
            check(|c| {
                c.save_image_max_age_days = 0;
                c.save_image_max_mb = 0;
            })
            .is_ok()
        );
        let old: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(old.save_image_max_age_days, 30);
    }

    #[test]
    fn validate_thread_and_concurrency_counts() {
        rejects(
            |c| c.intra_threads = 257,
            "intra_threads",
            "between 0 and 256",
        );
        rejects(
            |c| c.preprocess_threads = 10_000,
            "preprocess_threads",
            "between 0 and 256",
        );
        rejects(|c| c.inflight = 65, "inflight", "between 0 and 64");
        rejects(|c| c.ort_sessions = 17, "ort_sessions", "between 0 and 16");
        rejects(|c| c.gpu_index = 65, "gpu_index", "between 0 and 64");
        assert!(
            check(|c| {
                c.intra_threads = 256;
                c.preprocess_threads = 256;
                c.inflight = 64;
                c.ort_sessions = 16;
                c.gpu_index = 64;
            })
            .is_ok()
        );
        let mut c = three_models();
        c.models[0].inflight = Some(100);
        let err = format!("{:#}", c.validate().unwrap_err());
        assert!(
            err.contains("'a' inflight") && err.contains("0 and 64"),
            "{err}"
        );
    }

    #[test]
    fn validate_max_image_pixels() {
        rejects(
            |c| c.max_image_pixels = 0,
            "max_image_pixels",
            "between 1000 and 1000000000",
        );
        rejects(
            |c| c.max_image_pixels = 2_000_000_000,
            "max_image_pixels",
            "1000000000",
        );
        assert!(check(|c| c.max_image_pixels = 1_000).is_ok());
    }

    #[test]
    fn validate_paths() {
        rejects(|c| c.models_dir = PathBuf::new(), "models_dir", "empty");
        // An empty cache dir disables the cache.
        assert!(check(|c| c.cache_dir = String::new()).is_ok());
        let mut c = three_models();
        c.models[2].path = PathBuf::new();
        assert!(format!("{:#}", c.validate().unwrap_err()).contains("path"));
        // Duplicate names.
        let mut c = three_models();
        c.models[1].name = Some("A".into());
        assert!(format!("{:#}", c.validate().unwrap_err()).contains("duplicate"));
    }

    #[test]
    fn validate_devices() {
        rejects(|c| c.device = "xpu".into(), "device", "");
        assert!(check(|c| c.device = "ort:cuda:1".into()).is_ok());
        let mut c = three_models();
        c.models[0].device = Some("G P U".into());
        assert!(format!("{:#}", c.validate().unwrap_err()).contains("'a' device"));
    }

    #[test]
    fn validate_benchmark_section() {
        rejects(
            |c| c.benchmark.warmup = 1001,
            "benchmark.warmup",
            "0 and 1000",
        );
        rejects(
            |c| c.benchmark.repeat_per_image = 5000,
            "benchmark.repeat_per_image",
            "0 and 1000",
        );
        rejects(
            |c| c.benchmark.weights.accuracy = 2.0,
            "benchmark.weights.accuracy",
            "0 and 1",
        );
        rejects(
            |c| c.benchmark.weights.speed = f64::INFINITY,
            "benchmark.weights.speed",
            "0 and 1",
        );
        assert!(check(|c| c.benchmark.max_images_per_dataset = usize::MAX).is_ok());
    }

    // One test per validation rule of `Config::validate`, `validate_models` and `validate_config`
    // (PR-5 test gap): break exactly one field of a valid config, assert the error names it.

    /// `validate_config` on `three_models()` with `edit` applied: the error text.
    fn run_err(edit: impl FnOnce(&mut Config)) -> String {
        let mut c = three_models();
        edit(&mut c);
        format!("{:#}", validate_config(&c).unwrap_err())
    }

    #[test]
    fn rule_port_zero() {
        assert!(check(|c| c.port = 0).unwrap_err().contains("port:"));
    }

    #[test]
    fn rule_request_timeout_secs_out_of_range() {
        let e = check(|c| c.request_timeout_secs = 0).unwrap_err();
        assert!(e.contains("request_timeout_secs:"), "{e}");
    }

    #[test]
    fn rule_worker_queue_size_too_large() {
        let e = check(|c| c.worker_queue_size = 5000).unwrap_err();
        assert!(e.contains("worker_queue_size:"), "{e}");
    }

    #[test]
    fn rule_confidence_threshold_above_one() {
        let e = check(|c| c.confidence_threshold = 1.5).unwrap_err();
        assert!(e.contains("confidence_threshold:"), "{e}");
    }

    #[test]
    fn rule_nms_iou_not_finite() {
        let e = check(|c| c.nms_iou = f32::INFINITY).unwrap_err();
        assert!(e.contains("nms_iou:"), "{e}");
    }

    #[test]
    fn rule_intra_threads_too_large() {
        let e = check(|c| c.intra_threads = 300).unwrap_err();
        assert!(e.contains("intra_threads:"), "{e}");
    }

    #[test]
    fn rule_preprocess_threads_too_large() {
        let e = check(|c| c.preprocess_threads = 300).unwrap_err();
        assert!(e.contains("preprocess_threads:"), "{e}");
    }

    #[test]
    fn rule_inflight_too_large() {
        let e = check(|c| c.inflight = 65).unwrap_err();
        assert!(e.contains("inflight:"), "{e}");
    }

    #[test]
    fn rule_ort_sessions_too_large() {
        let e = check(|c| c.ort_sessions = 17).unwrap_err();
        assert!(e.contains("ort_sessions:"), "{e}");
    }

    #[test]
    fn rule_gpu_index_too_large() {
        let e = check(|c| c.gpu_index = 65).unwrap_err();
        assert!(e.contains("gpu_index:"), "{e}");
    }

    #[test]
    fn rule_max_image_pixels_too_small() {
        let e = check(|c| c.max_image_pixels = 1).unwrap_err();
        assert!(e.contains("max_image_pixels:"), "{e}");
    }

    #[test]
    fn rule_save_image_max_age_days_too_large() {
        let e = check(|c| c.save_image_max_age_days = 4000).unwrap_err();
        assert!(e.contains("save_image_max_age_days:"), "{e}");
    }

    #[test]
    fn rule_save_image_max_mb_too_large() {
        let e = check(|c| c.save_image_max_mb = 2_000_000).unwrap_err();
        assert!(e.contains("save_image_max_mb:"), "{e}");
    }

    #[test]
    fn rule_models_dir_empty() {
        let e = check(|c| c.models_dir = PathBuf::new()).unwrap_err();
        assert!(e.contains("models_dir:"), "{e}");
    }

    #[test]
    fn rule_device_unparsable() {
        let e = check(|c| c.device = "xpu".into()).unwrap_err();
        assert!(e.contains("device:"), "{e}");
    }

    #[test]
    fn rule_benchmark_warmup_too_large() {
        let e = check(|c| c.benchmark.warmup = 5000).unwrap_err();
        assert!(e.contains("benchmark.warmup:"), "{e}");
    }

    #[test]
    fn rule_benchmark_repeat_per_image_too_large() {
        let e = check(|c| c.benchmark.repeat_per_image = 5000).unwrap_err();
        assert!(e.contains("benchmark.repeat_per_image:"), "{e}");
    }

    #[test]
    fn rule_benchmark_weights_accuracy_above_one() {
        let e = check(|c| c.benchmark.weights.accuracy = 1.5).unwrap_err();
        assert!(e.contains("benchmark.weights.accuracy:"), "{e}");
    }

    #[test]
    fn rule_benchmark_weights_speed_negative() {
        let e = check(|c| c.benchmark.weights.speed = -0.1).unwrap_err();
        assert!(e.contains("benchmark.weights.speed:"), "{e}");
    }

    #[test]
    fn rule_models_path_empty() {
        let e = run_err(|c| c.models[0].path = PathBuf::new());
        assert!(e.contains("models:") && e.contains("`path`"), "{e}");
    }

    #[test]
    fn rule_models_name_duplicate() {
        let e = run_err(|c| c.models[1].name = Some("A".into()));
        assert!(e.contains("models:") && e.contains("duplicate"), "{e}");
    }

    #[test]
    fn rule_models_device_unparsable() {
        let e = run_err(|c| c.models[0].device = Some("xpu".into()));
        assert!(e.contains("'a' device"), "{e}");
    }

    #[test]
    fn rule_models_confidence_threshold_above_one() {
        let e = run_err(|c| c.models[0].confidence_threshold = Some(2.0));
        assert!(e.contains("'a' confidence_threshold"), "{e}");
    }

    #[test]
    fn rule_models_inflight_too_large() {
        let e = run_err(|c| c.models[0].inflight = Some(65));
        assert!(e.contains("'a' inflight"), "{e}");
    }

    #[test]
    fn rule_models_empty_list() {
        let e = validate_models(&[]).unwrap_err().to_string();
        assert!(e.contains("models: at least one model"), "{e}");
        let e = run_err(|c| c.models.clear());
        assert!(e.contains("models: at least one model"), "{e}");
    }

    #[test]
    fn rule_models_none_enabled() {
        let e = run_err(|c| c.models.iter_mut().for_each(|m| m.enabled = false));
        assert!(e.contains("models:") && e.contains("enabled"), "{e}");
    }

    #[test]
    fn rule_default_model_unknown() {
        let e = run_err(|c| c.default_model = Some("nope".into()));
        assert!(e.contains("default_model:"), "{e}");
    }

    #[test]
    fn rule_check_threshold_and_range_helpers() {
        assert!(
            check_threshold("x", -0.5)
                .unwrap_err()
                .to_string()
                .starts_with("x:")
        );
        assert!(
            check_range("y", 9u32, 0, 3)
                .unwrap_err()
                .to_string()
                .starts_with("y:")
        );
    }

    #[test]
    fn load_rejects_invalid_values_naming_the_key() {
        let dir = std::env::temp_dir().join(format!("bop-cfg-val-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.json");
        std::fs::write(&path, r#"{"confidence_threshold":5}"#).unwrap();
        let err = format!("{:#}", Config::load(&path).unwrap_err());
        assert!(
            err.contains("confidence_threshold: must be between 0 and 1, got 5"),
            "{err}"
        );
        std::fs::write(&path, r#"{"port":4000}"#).unwrap();
        assert_eq!(Config::load(&path).unwrap().port, 4000);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_is_atomic_and_leaves_no_temp_file() {
        let dir = std::env::temp_dir().join(format!("bop-cfg-save-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.json");
        std::fs::write(&path, "old").unwrap();
        let c = Config {
            port: 4321,
            ..Default::default()
        };
        c.save(&path).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            c.saved_text().unwrap()
        );
        assert!(!dir.join("c.json.tmp").exists());
        // A failed write (the target is a directory) keeps nothing behind.
        let bad = dir.join("sub");
        std::fs::create_dir_all(bad.join("x")).unwrap();
        assert!(c.save(&bad).is_err());
        assert!(!dir.join("sub.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn log_format_parses_and_names_the_key_on_error() {
        let c: Config = serde_json::from_str(r#"{"log_format":"json"}"#).unwrap();
        assert_eq!(c.log_format, LogFormat::Json);
        let c: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(c.log_format, LogFormat::Text);
        let err = serde_json::from_str::<Config>(r#"{"log_format":"xml"}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("log_format") && err.contains("xml"), "{err}");
        let text = serde_json::to_string(&Config::default()).unwrap();
        assert!(text.contains(r#""log_format":"text""#), "{text}");
        assert_eq!(" JSON ".parse::<LogFormat>().unwrap(), LogFormat::Json);
    }

    #[test]
    fn seed_default_model_only_fills_an_empty_list() {
        let mut c = Config::default();
        assert_eq!(c.seed_default_model().as_deref(), Some(DEFAULT_MODEL_NAME));
        assert_eq!(c.models.len(), 1);
        let m = &c.models[0];
        assert_eq!(m.path, Path::new("models").join("IPcam-general.onnx"));
        assert_eq!(m.family, ModelFamilyKind::Yolo5);
        assert!(m.enabled);
        // The seeded config is one the server runs with.
        assert!(validate_config(&c).is_ok());
        assert_eq!(c.seed_default_model(), None);
        assert_eq!(c.models.len(), 1);
    }

    /// Every config key (top level, model entry, `benchmark.*`) has a row `| `key` |` in the
    /// README's configuration tables.
    #[test]
    fn readme_documents_every_config_key() {
        let readme = include_str!("../README.md");
        let keys = |v: serde_json::Value| -> Vec<String> {
            match v {
                serde_json::Value::Object(m) => m.keys().cloned().collect(),
                _ => panic!("not an object"),
            }
        };
        let mut missing = Vec::new();
        let mut check = |key: String| {
            if !readme.contains(&format!("| `{key}` |")) {
                missing.push(key);
            }
        };
        let top = keys(serde_json::to_value(night_rules_config()).unwrap());
        assert!(top.contains(&"log_format".to_string()));
        assert!(top.contains(&"static_suppression".to_string()));
        for k in top {
            check(k);
        }
        // `Some` for the fields serialized only when set.
        let model = ModelConfig {
            inflight: Some(1),
            night_model: Some("night".into()),
            verify: Some(crate::verify::VerifyConfig::with_model("night")),
            ..night_rules_model()
        };
        let model_keys = keys(serde_json::to_value(model).unwrap());
        assert!(model_keys.contains(&"inflight".to_string()));
        for k in model_keys {
            check(k);
        }
        for k in keys(serde_json::to_value(BenchmarkConfig::default()).unwrap()) {
            check(format!("benchmark.{k}"));
        }
        assert!(
            missing.is_empty(),
            "README.md has no `| `key` |` row for: {missing:?}"
        );
    }

    /// A config with every night / IR rule set (globals and one model).
    fn night_rules_config() -> Config {
        Config {
            night_confidence_threshold: Some(0.6),
            label_thresholds: BTreeMap::from([("car".to_string(), 0.7)]),
            night_label_thresholds: BTreeMap::from([("person".to_string(), 0.65)]),
            person_aspect: Some([0.2, 0.9]),
            min_box_area_frac: Some(0.001),
            suppress_overlaps: vec![OverlapRule {
                drop: "person".into(),
                if_overlaps: "potted plant".into(),
                iou: 0.45,
                night_only: true,
            }],
            static_suppression: StaticSuppression {
                enabled: true,
                ..Default::default()
            },
            preprocess: Some(Preprocess::Gamma(0.8)),
            night_preprocess: Some(Preprocess::CLAHE),
            verify: Some(crate::verify::VerifyConfig::with_model("m")),
            models: vec![night_rules_model()],
            ..Config::default()
        }
    }

    fn night_rules_model() -> ModelConfig {
        ModelConfig {
            name: Some("m".into()),
            path: "m.onnx".into(),
            night_confidence_threshold: Some(0.7),
            label_thresholds: Some(BTreeMap::from([("person".to_string(), 0.55)])),
            night_label_thresholds: Some(BTreeMap::new()),
            person_aspect: Some([0.25, 0.8]),
            min_box_area_frac: Some(0.0005),
            suppress_overlaps: Some(Vec::new()),
            static_suppression: Some(StaticSuppression::default()),
            preprocess: Some(Preprocess::None),
            night_preprocess: Some(Preprocess::Clahe {
                clip: 3.0,
                tiles: 4,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn old_config_without_night_rules_round_trips_unchanged() {
        let mut c = Config::default();
        c.models.push(ModelConfig {
            name: Some("a".into()),
            path: "models/a.onnx".into(),
            ..Default::default()
        });
        let text = c.saved_text().unwrap();
        for key in [
            "night_confidence_threshold",
            "label_thresholds",
            "night_label_thresholds",
            "person_aspect",
            "min_box_area_frac",
            "suppress_overlaps",
            "static_suppression",
            "\"preprocess\"",
            "night_preprocess",
            "verify",
        ] {
            assert!(!text.contains(key), "{key} written by default: {text}");
        }
        let back: Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        assert_eq!(back.saved_text().unwrap(), text);
    }

    #[test]
    fn night_rules_round_trip_with_clean_numbers() {
        let c = night_rules_config();
        c.validate().unwrap();
        let text = c.saved_text().unwrap();
        let back: Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        assert_eq!(back.saved_text().unwrap(), text);
        // Through serde_json::Value (what config_merge compares) the numbers stay as typed.
        let v = serde_json::to_value(&c).unwrap().to_string();
        for want in [
            r#""night_confidence_threshold":0.6"#,
            r#""person_aspect":[0.2,0.9]"#,
            r#""min_box_area_frac":0.001"#,
            r#""night_label_thresholds":{"person":0.65}"#,
            r#""iou":0.45"#,
            r#""iou":0.85"#,
            r#""preprocess":"gamma:0.8""#,
            r#""night_preprocess":"clahe""#,
            r#""night_preprocess":"clahe:3,4""#,
        ] {
            assert!(v.contains(want), "{want} not in {v}");
        }
        // A static_suppression with only `enabled` gets the defaults.
        let c: Config = serde_json::from_str(r#"{"static_suppression":{"enabled":true}}"#).unwrap();
        assert!(c.static_suppression.enabled);
        assert_eq!(c.static_suppression.hits, 5);
    }

    #[test]
    fn validate_verify_references() {
        use crate::verify::VerifyConfig;
        let entry = |name: &str, verify: Option<VerifyConfig>| ModelConfig {
            name: Some(name.into()),
            path: format!("{name}.onnx").into(),
            verify,
            ..Default::default()
        };
        let check = |c: Config| c.validate().map_err(|e| format!("{e:#}"));
        let v = |m: &str| Some(VerifyConfig::with_model(m));
        let ok = Config {
            models: vec![entry("a", v("B.onnx")), entry("b", None)],
            ..Config::default()
        };
        check(ok.clone()).unwrap();
        // Unknown, itself, out of range.
        let e = check(Config {
            models: vec![entry("a", v("c")), entry("b", None)],
            ..Config::default()
        })
        .unwrap_err();
        assert!(
            e.contains("'a' verify.model: 'c' is not a configured model"),
            "{e}"
        );
        let e = check(Config {
            models: vec![entry("a", v("A"))],
            ..Config::default()
        })
        .unwrap_err();
        assert!(
            e.contains("'a' verify.model: names the model itself"),
            "{e}"
        );
        let bad = VerifyConfig {
            max_crops: 0,
            ..VerifyConfig::with_model("b")
        };
        let e = check(Config {
            models: vec![entry("a", Some(bad)), entry("b", None)],
            ..Config::default()
        })
        .unwrap_err();
        assert!(e.contains("models: 'a' verify.max_crops"), "{e}");
        // A disabled entry is not checked; verifiers may verify each other (checks never
        // cascade).
        let off = VerifyConfig {
            enabled: false,
            ..VerifyConfig::default()
        };
        check(Config {
            models: vec![
                entry("a", Some(off)),
                entry("b", v("a")),
                entry("c", v("a")),
            ],
            ..ok.clone()
        })
        .unwrap();
        check(Config {
            models: vec![entry("a", v("b")), entry("b", v("a"))],
            ..Config::default()
        })
        .unwrap();
        // The global default names a configured model (it never verifies that model itself).
        check(Config {
            verify: v("b"),
            ..ok.clone()
        })
        .unwrap();
        let e = check(Config {
            verify: v("zzz"),
            ..ok.clone()
        })
        .unwrap_err();
        assert!(
            e.contains("verify.model: 'zzz' is not a configured model"),
            "{e}"
        );
        let e = check(Config {
            verify: Some(VerifyConfig {
                pad: -1.0,
                ..VerifyConfig::with_model("b")
            }),
            ..ok
        })
        .unwrap_err();
        assert!(e.starts_with("verify.pad"), "{e}");
    }

    #[test]
    fn verify_round_trips_and_resolves_per_model() {
        use crate::verify::VerifyConfig;
        let text = r#"{"verify":{"model":"b","aliases":{"vehicle":"car"}},"models":[
            {"name":"a","path":"a.onnx"},
            {"name":"b","path":"b.onnx"},
            {"name":"c","path":"c.onnx","verify":{"enabled":false}},
            {"name":"d","path":"d.onnx","verify":{"model":"a","mode":"frame","night_only":false}}]}"#;
        let c: Config = serde_json::from_str(text).unwrap();
        c.validate().unwrap();
        let back: Config = serde_json::from_str(&c.saved_text().unwrap()).unwrap();
        assert_eq!(back, c);
        let live = |i: usize| crate::worker::verify_settings(&c, &c.models[i]);
        let a = live(0).unwrap();
        assert_eq!(a.model, "b");
        assert_eq!(a.aliases.get("vehicle").map(String::as_str), Some("car"));
        assert_eq!(live(1), None, "the verifier is never verified by itself");
        assert_eq!(live(2), None, "turned off for this model");
        let d = live(3).unwrap();
        assert_eq!(
            (d.model.as_str(), d.mode, d.night_only),
            ("a", crate::verify::VerifyMode::Frame, false)
        );
        assert_eq!(d.labels, VerifyConfig::default().labels);
    }

    #[test]
    fn preprocess_modes_resolve_per_field() {
        let mut c = Config::default();
        let mut m = ModelConfig::default();
        assert!(c.preprocess_modes(&m).is_noop());
        c.preprocess = Some(Preprocess::Grayscale);
        c.night_preprocess = Some(Preprocess::CLAHE);
        let g = c.preprocess_modes(&m);
        assert_eq!(g.preprocess, Preprocess::Grayscale);
        assert_eq!(g.night_preprocess, Some(Preprocess::CLAHE));
        // The model's own value wins per field; `none` turns a global mode off.
        m.preprocess = Some(Preprocess::None);
        let own = c.preprocess_modes(&m);
        assert_eq!(own.preprocess, Preprocess::None);
        assert_eq!(own.night_preprocess, Some(Preprocess::CLAHE));
        m.night_preprocess = Some(Preprocess::Gamma(0.6));
        assert_eq!(
            c.preprocess_modes(&m).night_preprocess,
            Some(Preprocess::Gamma(0.6))
        );
        // A bad mode is a parse error that lists the accepted forms.
        let e = serde_json::from_str::<Config>(r#"{"preprocess":"sharpen"}"#)
            .unwrap_err()
            .to_string();
        assert!(e.contains("not a mode") && e.contains("clahe"), "{e}");
    }

    #[test]
    fn validate_night_rules() {
        let night = |f: fn(&mut Config)| {
            let mut c = Config::default();
            f(&mut c);
            c.validate().map_err(|e| format!("{e:#}"))
        };
        assert!(night(|c| *c = night_rules_config()).is_ok());
        let e = night(|c| c.night_confidence_threshold = Some(1.5)).unwrap_err();
        assert!(
            e.contains("night_confidence_threshold: must be between 0 and 1"),
            "{e}"
        );
        let e = night(|c| {
            c.label_thresholds.insert("person".into(), -0.1);
        })
        .unwrap_err();
        assert!(e.contains("label_thresholds.person"), "{e}");
        let e = night(|c| {
            c.night_label_thresholds.insert(" ".into(), 0.5);
        })
        .unwrap_err();
        assert!(
            e.contains("night_label_thresholds: has an empty label"),
            "{e}"
        );
        for bad in [
            [0.9, 0.2],
            [0.5, 0.5],
            [-0.1, 0.9],
            [0.2, 50.0],
            [f32::NAN, 1.0],
        ] {
            let c = Config {
                person_aspect: Some(bad),
                ..Config::default()
            };
            let e = format!("{:#}", c.validate().unwrap_err());
            assert!(e.contains("person_aspect: must be [min, max]"), "{e}");
        }
        let e = night(|c| c.min_box_area_frac = Some(2.0)).unwrap_err();
        assert!(e.contains("min_box_area_frac"), "{e}");
        let rule = |drop: &str, iou: f32| OverlapRule {
            drop: drop.into(),
            if_overlaps: "potted plant".into(),
            iou,
            night_only: false,
        };
        let mut c = Config {
            suppress_overlaps: vec![rule("person", 0.5), rule("", 0.5)],
            ..Config::default()
        };
        let e = format!("{:#}", c.validate().unwrap_err());
        assert!(e.contains("suppress_overlaps[1]"), "{e}");
        c.suppress_overlaps = vec![rule("person", 0.0)];
        let e = format!("{:#}", c.validate().unwrap_err());
        assert!(
            e.contains("suppress_overlaps[0].iou: must be above 0"),
            "{e}"
        );
        let e = night(|c| c.static_suppression.hits = 0).unwrap_err();
        assert!(e.contains("static_suppression.hits"), "{e}");
        // Per model: the error names the model.
        let mut c = Config::default();
        c.models.push(ModelConfig {
            person_aspect: Some([1.0, 0.5]),
            ..night_rules_model()
        });
        let e = format!("{:#}", c.validate().unwrap_err());
        assert!(e.contains("models: 'm' person_aspect"), "{e}");
        let mut c = Config::default();
        c.models.push(ModelConfig {
            static_suppression: Some(StaticSuppression {
                window_minutes: 0,
                ..Default::default()
            }),
            ..night_rules_model()
        });
        let e = format!("{:#}", c.validate().unwrap_err());
        assert!(
            e.contains("models: 'm' static_suppression.window_minutes"),
            "{e}"
        );
    }
}
