//! Service configuration: a JSON file next to the executable, merged with CLI flags.
//! Merge rule (copied from blue-onyx): a CLI value overrides the file value only when it differs
//! from the built-in default, and the merged result is written back to the file.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::model::ModelFamilyKind;

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
        match std::fs::rename(&old, &path) {
            Ok(()) => tracing::info!("renamed config {} -> {}", old.display(), path.display()),
            Err(e) => {
                tracing::warn!(
                    "could not rename config {}: {e}; using it as is",
                    old.display()
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

impl std::str::FromStr for LogLevel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "trace" => LogLevel::Trace,
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" | "warning" => LogLevel::Warn,
            "error" => LogLevel::Error,
            other => anyhow::bail!("unknown log level '{other}'"),
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
    pub log_level: LogLevel,
    pub log_path: Option<PathBuf>,
    /// Save annotated images here (optional).
    pub save_image_path: Option<PathBuf>,
    pub save_ref_image: bool,
    /// CPU inference threads (0 = OpenVINO default).
    pub intra_threads: usize,
    /// Decode + preprocess threads shared by every model (0 = from the hardware profile:
    /// `clamp(cores / 2, 1, 8)`, at most 2 when every model runs on the CPU). Restart.
    pub preprocess_threads: usize,
    /// Concurrent inferences per model (0 = from the hardware profile and the model's device,
    /// see [`crate::profile::Profile::tuning_for`]); a model's own `inflight` wins. Restart.
    pub inflight: usize,
    /// ONNX Runtime sessions per model, one inference in flight each (0 = from the hardware
    /// profile: 2 on a high-spec machine with an NVIDIA GPU, else 1). Each session holds its own
    /// copy of the model. Restart.
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
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            datasets: crate::benchmark::images::KNOWN_SET_IDS
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
            log_level: LogLevel::Info,
            log_path: None,
            save_image_path: None,
            save_ref_image: false,
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
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mut config: Config = serde_json::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        config.normalize();
        config
            .validate()
            .with_context(|| format!("invalid config {}", path.display()))?;
        Ok(config)
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
        anyhow::bail!("{field}: must be between {lo} and {hi}, got {v}");
    }
    Ok(())
}

/// The one rule for confidence thresholds and `nms_iou` (global and per model): a finite
/// number in `0..=1`. 0 is allowed (report every box).
pub fn check_threshold(field: &str, t: f32) -> Result<()> {
    if !(t.is_finite() && (0.0..=1.0).contains(&t)) {
        anyhow::bail!("{field}: must be between 0 and 1, got {t}");
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
            bail!("models: every entry needs a `path`");
        }
        let key = crate::registry::normalize_name(&m.effective_name());
        if !seen.insert(key.clone()) {
            bail!("models: duplicate model name '{key}'; set a distinct `name` for each entry");
        }
        if let Some(d) = &m.device {
            crate::backend::spec::parse(d)
                .map_err(|e| anyhow::anyhow!("models: '{key}' device: {e}"))?;
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
    }
    Ok(())
}

/// The models list on its own: at least one entry, and [`validate_model_entries`].
pub fn validate_models(models: &[ModelConfig]) -> Result<()> {
    if models.is_empty() {
        anyhow::bail!("models: at least one model is required");
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
        if self.models_dir.as_os_str().is_empty() {
            bail!("models_dir: must not be empty");
        }
        crate::backend::spec::parse(&self.device).map_err(|e| anyhow::anyhow!("device: {e}"))?;
        validate_model_entries(&self.models)?;
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

/// Checks of a config the server runs with: [`Config::validate`], at least one model, at least
/// one enabled model and a `default_model` that names a configured model.
pub fn validate_config(c: &Config) -> Result<()> {
    use anyhow::bail;
    c.validate()?;
    if c.models.is_empty() {
        bail!("models: at least one model is required");
    }
    if !c.models.iter().any(|m| m.enabled) {
        bail!("models: at least one model must be enabled");
    }
    if let Some(name) = &c.default_model
        && c.default_model_index().is_none()
    {
        bail!("default_model: '{name}' is not one of the configured models");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
}
