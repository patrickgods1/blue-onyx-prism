//! Hardware profiles (docs/REVIEW_PLAN.md, phase 2.5): one binary, with the concurrency tunables
//! of the pipelined worker defaulting from a profile derived from facts the process already has
//! (logical cores, physical memory, the detected GPUs). Explicit config values always win; the
//! profile only fills `0` / `auto`.
//!
//! Everything here is pure except [`Facts::detect`]. The classes:
//! - **Low**: <= 6 logical cores, or <= 8 GB of memory, or no GPU / only an integrated GPU on
//!   x86 (the i5-8500 + UHD 630 design target lands here).
//! - **High**: >= 16 GB and either >= 12 cores with a discrete GPU of >= 6 GB VRAM, or an Apple
//!   GPU with >= 8 cores.
//! - **Mid**: everything else.
//!
//! What a class changes is deliberately small: decode + preprocess threads are
//! `clamp(cores / 2, 1, 8)` for every class (2 at most when every model runs on the CPU), and
//! concurrent inference depends on the *device* a model runs on ([`Profile::tuning_for`]),
//! not on the machine class. The class picks the recommended model sizes. A discrete GPU of at
//! least [`ORT_TWO_SESSIONS_VRAM_MB`] with at least [`ORT_TWO_SESSIONS_MEMORY_GB`] of memory (or
//! unknown memory) gets a second ONNX Runtime session per model whatever the class: one session
//! leaves the GPU idle while the host decodes, preprocesses and copies (a 6-core box with a
//! GTX 1070 Ti is Low yet stays at ~60% GPU with DirectML).

use crate::backend::detect::{GpuVendor, HardwareInfo};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Upper bound of concurrent inferences per model (slots of the worker).
pub const MAX_INFLIGHT: usize = 8;
/// Upper bound of the decode + preprocess pool.
pub const MAX_PREPROCESS_THREADS: usize = 8;
/// Pool size when every model runs on the CPU (CPU inference already uses all cores but one).
pub const CPU_ONLY_PREPROCESS_THREADS: usize = 2;
/// Logical cores from which CPU inference is compiled for throughput (several streams).
pub const CPU_THROUGHPUT_CORES: usize = 12;
/// In flight on an integrated GPU: overlap the host side of one frame with the GPU on the other.
pub const IGPU_INFLIGHT: usize = 2;
/// In flight on a discrete OpenVINO GPU (Intel Arc), clamped to its `NUM_STREAMS`.
pub const DGPU_INFLIGHT: usize = 4;

/// VRAM from which a discrete GPU gets two ONNX Runtime sessions per model (each holds a copy).
pub const ORT_TWO_SESSIONS_VRAM_MB: u64 = 4096;
/// Physical memory needed for two ONNX Runtime sessions per model (unknown counts as enough).
pub const ORT_TWO_SESSIONS_MEMORY_GB: f64 = 8.0;

/// What kind of GPU the machine has (the most capable one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum GpuClass {
    None,
    Integrated,
    Discrete { vram_mb: u64 },
    Apple,
}

impl GpuClass {
    /// `none`, `integrated`, `discrete`, `apple`.
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuClass::None => "none",
            GpuClass::Integrated => "integrated",
            GpuClass::Discrete { .. } => "discrete",
            GpuClass::Apple => "apple",
        }
    }
}

/// The facts a profile is derived from.
#[derive(Debug, Clone, PartialEq)]
pub struct Facts {
    pub logical_cores: usize,
    /// Total physical memory in GiB (None when unknown).
    pub memory_gb: Option<f64>,
    pub gpu: GpuClass,
    /// `std::env::consts::ARCH`: "x86_64", "aarch64", ...
    pub arch: String,
    /// Name of the GPU `gpu` describes, e.g. "Intel(R) UHD Graphics 630".
    pub gpu_name: Option<String>,
    /// Vendor of that GPU.
    pub gpu_vendor: Option<GpuVendor>,
}

impl Facts {
    /// Facts from already known values. The GPU is the discrete one with the most VRAM, else
    /// an Apple GPU, else the first integrated one.
    pub fn from_hardware(logical_cores: usize, memory_gb: Option<f64>, hw: &HardwareInfo) -> Self {
        let discrete = hw
            .gpus
            .iter()
            .filter(|g| g.discrete && g.vendor != GpuVendor::Apple)
            .max_by_key(|g| g.vram_mb);
        let apple = hw.gpus.iter().find(|g| g.vendor == GpuVendor::Apple);
        let integrated = hw.gpus.iter().find(|g| !g.discrete);
        let (gpu, adapter) = if let Some(g) = discrete {
            (GpuClass::Discrete { vram_mb: g.vram_mb }, Some(g))
        } else if let Some(g) = apple {
            (GpuClass::Apple, Some(g))
        } else if let Some(g) = integrated {
            (GpuClass::Integrated, Some(g))
        } else {
            (GpuClass::None, None)
        };
        Self {
            logical_cores: logical_cores.max(1),
            memory_gb,
            gpu,
            arch: hw.arch.clone(),
            gpu_name: adapter.map(|g| g.name.clone()),
            gpu_vendor: adapter.map(|g| g.vendor),
        }
    }

    /// This machine: `num_cpus`, [`crate::system_info::total_memory_gb`] and the detected GPUs
    /// ([`crate::backend::detect::hardware`]). The only impure function of this module; computed
    /// once per process (memory on Windows takes a PowerShell call).
    pub fn detect() -> Self {
        static FACTS: OnceLock<Facts> = OnceLock::new();
        FACTS
            .get_or_init(|| {
                Facts::from_hardware(
                    num_cpus::get(),
                    crate::system_info::total_memory_gb(),
                    crate::backend::detect::hardware(),
                )
            })
            .clone()
    }

    fn is_x86(&self) -> bool {
        matches!(self.arch.as_str(), "x86_64" | "x86")
    }
}

/// Machine class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HardwareClass {
    Low,
    Mid,
    High,
}

impl HardwareClass {
    /// `low`, `mid`, `high`.
    pub fn as_str(self) -> &'static str {
        match self {
            HardwareClass::Low => "low",
            HardwareClass::Mid => "mid",
            HardwareClass::High => "high",
        }
    }
}

/// Config `hardware_profile`: `auto` (classify) or a forced class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HardwareProfile {
    #[default]
    Auto,
    Low,
    Mid,
    High,
}

impl HardwareProfile {
    /// The forced class (None = auto).
    pub fn forced(self) -> Option<HardwareClass> {
        match self {
            HardwareProfile::Auto => None,
            HardwareProfile::Low => Some(HardwareClass::Low),
            HardwareProfile::Mid => Some(HardwareClass::Mid),
            HardwareProfile::High => Some(HardwareClass::High),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            HardwareProfile::Auto => "auto",
            HardwareProfile::Low => "low",
            HardwareProfile::Mid => "mid",
            HardwareProfile::High => "high",
        }
    }
}

impl std::str::FromStr for HardwareProfile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Ok(HardwareProfile::Auto),
            "low" => Ok(HardwareProfile::Low),
            "mid" => Ok(HardwareProfile::Mid),
            "high" => Ok(HardwareProfile::High),
            other => Err(format!("'{other}' is not one of auto, low, mid, high")),
        }
    }
}

/// Model size of a catalog model, from its name suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    N,
    S,
    M,
    L,
    X,
}

impl ModelSize {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelSize::N => "n",
            ModelSize::S => "s",
            ModelSize::M => "m",
            ModelSize::L => "l",
            ModelSize::X => "x",
        }
    }

    fn from_suffix(s: &str) -> Option<Self> {
        Some(match s {
            "n" | "t" | "nano" | "atto" | "femto" | "pico" => ModelSize::N,
            "s" | "small" => ModelSize::S,
            "m" | "ms" | "b" | "medium" | "base" => ModelSize::M,
            "l" | "c" | "large" => ModelSize::L,
            "x" | "e" | "xlarge" => ModelSize::X,
            _ => return None,
        })
    }
}

/// Size of a catalog model from its name: the part after the last `-` (`rt-detrv2-s`,
/// `rfdetr-medium`, `deimv2-atto`), or the letters after the version digits of an Ultralytics
/// name (`yolov8n`, `yolo11x`, `yolov9c`). Names without a size (the IPcam YOLOv5 models,
/// `delivery`, `package`) count as `s` when `yolo5` is true. Pure.
pub fn model_size(name: &str, yolo5: bool) -> Option<ModelSize> {
    let lower = name.trim().to_ascii_lowercase();
    let stem = lower
        .strip_suffix(".onnx")
        .or_else(|| lower.strip_suffix(".xml"))
        .unwrap_or(&lower);
    let found = if let Some((_, last)) = stem.rsplit_once('-') {
        ModelSize::from_suffix(last)
    } else if stem.starts_with("yolo") {
        let digits_end = stem.rfind(|c: char| c.is_ascii_digit()).map(|i| i + 1);
        digits_end.and_then(|i| ModelSize::from_suffix(&stem[i..]))
    } else {
        None
    };
    found.or(yolo5.then_some(ModelSize::S))
}

/// OpenVINO performance hint a model is compiled with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PerfHint {
    /// One stream, the lowest latency per request (the default).
    #[default]
    Latency,
    /// Several streams so several requests run at once.
    Throughput,
}

impl PerfHint {
    /// `LATENCY` / `THROUGHPUT` (the OpenVINO property value).
    pub fn as_str(self) -> &'static str {
        match self {
            PerfHint::Latency => "LATENCY",
            PerfHint::Throughput => "THROUGHPUT",
        }
    }
}

/// The kind of device a model runs on, for its concurrency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    /// OpenVINO CPU.
    Cpu,
    /// OpenVINO GPU on an integrated GPU.
    IntegratedGpu,
    /// OpenVINO GPU on a discrete GPU (Intel Arc).
    DiscreteGpu,
    /// OpenVINO NPU.
    Npu,
    /// ONNX Runtime; `gpu` for every execution provider but the CPU.
    Ort { gpu: bool },
}

/// How a model is compiled and how many inferences it keeps in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceTuning {
    pub hint: PerfHint,
    /// Concurrent inferences (worker slots). For ONNX Runtime one per session.
    pub inflight: usize,
    /// Clamp `inflight` to the `NUM_STREAMS` OpenVINO reports after compiling.
    pub clamp_to_streams: bool,
    /// ONNX Runtime sessions of the model (1 for OpenVINO).
    pub ort_sessions: usize,
}

impl DeviceTuning {
    /// One inference at a time with the latency hint (the benchmark's serial setup).
    pub const SERIAL: DeviceTuning = DeviceTuning {
        hint: PerfHint::Latency,
        inflight: 1,
        clamp_to_streams: false,
        ort_sessions: 1,
    };

    /// `inflight` after compiling with `streams` reported (None: not readable).
    pub fn effective_inflight(&self, streams: Option<usize>) -> usize {
        let n = match streams {
            Some(s) if self.clamp_to_streams => self.inflight.min(s.max(1)),
            _ => self.inflight,
        };
        n.clamp(1, MAX_INFLIGHT)
    }
}

/// Explicit concurrency settings from the config (0 = from the profile).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Overrides {
    pub inflight: usize,
    pub ort_sessions: usize,
}

/// The profile of this machine.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Profile {
    pub class: HardwareClass,
    /// The class was forced by `hardware_profile`.
    pub forced: bool,
    #[serde(skip)]
    pub facts: Facts,
    /// Decode + preprocess threads before the CPU-only cap ([`Self::preprocess_threads_for`]).
    pub preprocess_threads: usize,
    /// In flight on the device this machine most likely runs on ([`Self::primary_device`]).
    pub inflight: usize,
    /// ONNX Runtime sessions per model on a GPU execution provider.
    pub ort_sessions: usize,
    /// Model sizes recommended for this class (a hint; the benchmark decides).
    #[serde(rename = "recommendedSizes")]
    pub model_sizes: &'static [ModelSize],
    /// Why: "6 logical cores", "16 GB memory", "integrated GPU (Intel UHD 630)".
    pub reasons: Vec<String>,
}

const LOW_SIZES: &[ModelSize] = &[ModelSize::N, ModelSize::S];
const MID_SIZES: &[ModelSize] = &[ModelSize::S, ModelSize::M];
const HIGH_SIZES: &[ModelSize] = &[ModelSize::M, ModelSize::L, ModelSize::X];

fn gb_text(gb: f64) -> String {
    // Whole gigabytes; a negative or NaN input shows as 0.
    format!("{:.0} GB", gb.round().max(0.0))
}

/// Classify `facts` (see the module docs).
pub fn classify(facts: &Facts) -> Profile {
    classify_as(facts, None)
}

/// [`classify`], or `forced` when set (`hardware_profile`).
pub fn classify_as(facts: &Facts, forced: Option<HardwareClass>) -> Profile {
    let cores = facts.logical_cores.max(1);
    let mem = facts.memory_gb;
    let mut reasons = vec![format!("{cores} logical cores")];
    match mem {
        Some(gb) => reasons.push(format!("{} memory", gb_text(gb))),
        None => reasons.push("memory unknown".to_string()),
    }
    let name = facts.gpu_name.as_deref().unwrap_or("unknown");
    reasons.push(match facts.gpu {
        GpuClass::None => "no GPU".to_string(),
        GpuClass::Integrated => format!("integrated GPU ({name})"),
        GpuClass::Discrete { vram_mb } if vram_mb > 0 => {
            format!("discrete GPU ({name}, {} GB VRAM)", (vram_mb + 512) / 1024)
        }
        GpuClass::Discrete { .. } => format!("discrete GPU ({name}, VRAM unknown)"),
        GpuClass::Apple => format!("Apple GPU ({name})"),
    });

    let small_gpu = matches!(facts.gpu, GpuClass::None | GpuClass::Integrated) && facts.is_x86();
    let low = cores <= 6 || mem.is_some_and(|g| g <= 8.0) || small_gpu;
    let big_mem = mem.is_some_and(|g| g >= 16.0);
    let high = big_mem
        && match facts.gpu {
            GpuClass::Discrete { vram_mb } => cores >= 12 && vram_mb >= 6 * 1024,
            GpuClass::Apple => cores >= 8,
            _ => false,
        };
    let detected = if low {
        HardwareClass::Low
    } else if high {
        HardwareClass::High
    } else {
        HardwareClass::Mid
    };
    let class = forced.unwrap_or(detected);
    if forced.is_some_and(|f| f != detected) {
        reasons.push(format!(
            "forced by hardware_profile (detected: {})",
            detected.as_str()
        ));
    }
    let model_sizes = match class {
        HardwareClass::Low => LOW_SIZES,
        HardwareClass::Mid => MID_SIZES,
        HardwareClass::High => HIGH_SIZES,
    };
    // Deliberately not gated on the class (see the module doc).
    let big_dgpu =
        matches!(facts.gpu, GpuClass::Discrete { vram_mb } if vram_mb >= ORT_TWO_SESSIONS_VRAM_MB);
    let enough_memory = facts
        .memory_gb
        .is_none_or(|g| g >= ORT_TWO_SESSIONS_MEMORY_GB);
    let ort_sessions = if big_dgpu && enough_memory { 2 } else { 1 };
    let mut p = Profile {
        class,
        forced: forced.is_some(),
        facts: facts.clone(),
        preprocess_threads: (cores / 2).clamp(1, MAX_PREPROCESS_THREADS),
        inflight: 1,
        ort_sessions,
        model_sizes,
        reasons,
    };
    p.inflight = p.tuning_for(p.primary_device()).inflight;
    p
}

impl Profile {
    /// The profile of this machine for `setting` (`hardware_profile` of the config).
    pub fn current(setting: HardwareProfile) -> Self {
        classify_as(&Facts::detect(), setting.forced())
    }

    /// Decode + preprocess threads: [`Self::preprocess_threads`], at most
    /// [`CPU_ONLY_PREPROCESS_THREADS`] when every model runs on the CPU.
    pub fn preprocess_threads_for(&self, all_cpu: bool) -> usize {
        if all_cpu {
            self.preprocess_threads.min(CPU_ONLY_PREPROCESS_THREADS)
        } else {
            self.preprocess_threads
        }
    }

    /// Compile hint and concurrency for a model on `dev`:
    /// - integrated GPU: LATENCY, 2 in flight;
    /// - discrete OpenVINO GPU: THROUGHPUT, 4 in flight clamped to `NUM_STREAMS`;
    /// - CPU on >= 12 logical cores: THROUGHPUT, in flight = `NUM_STREAMS` (at most
    ///   [`MAX_INFLIGHT`]); fewer cores: LATENCY, 1 (one request already uses every core);
    /// - NPU: LATENCY, 1;
    /// - ONNX Runtime: one in flight per session; [`Self::ort_sessions`] sessions on a GPU
    ///   provider, 1 on the CPU.
    pub fn tuning_for(&self, dev: DeviceClass) -> DeviceTuning {
        let serial = DeviceTuning::SERIAL;
        match dev {
            DeviceClass::IntegratedGpu => DeviceTuning {
                inflight: IGPU_INFLIGHT,
                ..serial
            },
            DeviceClass::DiscreteGpu => DeviceTuning {
                hint: PerfHint::Throughput,
                inflight: DGPU_INFLIGHT,
                clamp_to_streams: true,
                ..serial
            },
            DeviceClass::Cpu if self.facts.logical_cores >= CPU_THROUGHPUT_CORES => DeviceTuning {
                hint: PerfHint::Throughput,
                inflight: MAX_INFLIGHT,
                clamp_to_streams: true,
                ..serial
            },
            DeviceClass::Cpu | DeviceClass::Npu => serial,
            DeviceClass::Ort { gpu } => {
                let sessions = if gpu { self.ort_sessions } else { 1 };
                DeviceTuning {
                    inflight: sessions,
                    ort_sessions: sessions,
                    ..serial
                }
            }
        }
    }

    /// [`Self::tuning_for`] with the explicit config values applied: `inflight` (model, else
    /// global; 0 = profile) replaces the count and is not clamped to the streams; for ONNX
    /// Runtime `ort_sessions` (0 = profile) sets the sessions and an explicit `inflight` caps
    /// them.
    pub fn tuning_with(&self, dev: DeviceClass, o: Overrides) -> DeviceTuning {
        let mut t = self.tuning_for(dev);
        if let DeviceClass::Ort { .. } = dev {
            if o.ort_sessions > 0 {
                t.ort_sessions = o.ort_sessions.min(MAX_INFLIGHT);
            }
            if o.inflight > 0 {
                t.ort_sessions = t.ort_sessions.min(o.inflight);
            }
            t.inflight = t.ort_sessions;
        } else if o.inflight > 0 {
            t.inflight = o.inflight.min(MAX_INFLIGHT);
            t.clamp_to_streams = false;
        }
        t
    }

    /// The device class this machine's `auto` most likely picks, for the summary: an Intel
    /// discrete GPU (OpenVINO), another discrete or Apple GPU (ONNX Runtime), an integrated GPU
    /// on x86 (OpenVINO), else the CPU.
    pub fn primary_device(&self) -> DeviceClass {
        match self.facts.gpu {
            GpuClass::Discrete { .. } if self.facts.gpu_vendor == Some(GpuVendor::Intel) => {
                DeviceClass::DiscreteGpu
            }
            GpuClass::Discrete { .. } | GpuClass::Apple => DeviceClass::Ort { gpu: true },
            GpuClass::Integrated if self.facts.is_x86() => DeviceClass::IntegratedGpu,
            _ => DeviceClass::Cpu,
        }
    }

    /// The recommended sizes as text: "n/s".
    pub fn sizes_text(&self) -> String {
        self.model_sizes
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("/")
    }

    /// One line for the welcome page and `list-devices`: "low-spec: 6 logical cores, 16 GB
    /// memory, integrated GPU (Intel(R) UHD Graphics 630) -> 3 preprocess threads, 2 in flight,
    /// recommended sizes n/s".
    pub fn summary(&self, preprocess_threads: usize) -> String {
        let inflight = match self.tuning_for(self.primary_device()) {
            t if t.clamp_to_streams => format!("up to {} in flight (NUM_STREAMS)", t.inflight),
            t => format!("{} in flight", t.inflight),
        };
        format!(
            "{}-spec{}: {} \u{2192} {preprocess_threads} preprocess thread{}, {inflight}, recommended sizes {}",
            self.class.as_str(),
            if self.forced { " (forced)" } else { "" },
            self.reasons.join(", "),
            if preprocess_threads == 1 { "" } else { "s" },
            self.sizes_text()
        )
    }

    /// The `profile` object of `GET /v1/devices`.
    pub fn to_json(&self, preprocess_threads: usize) -> serde_json::Value {
        serde_json::json!({
            "class": self.class,
            "forced": self.forced,
            "cores": self.facts.logical_cores,
            "memoryGb": self.facts.memory_gb.map(|g| (g * 10.0).round() / 10.0),
            "gpuClass": self.facts.gpu.as_str(),
            "gpuName": self.facts.gpu_name,
            "preprocessThreads": preprocess_threads,
            "inflight": self.inflight,
            "ortSessions": self.ort_sessions,
            "recommendedSizes": self.model_sizes,
            "reasons": self.reasons,
            "summary": self.summary(preprocess_threads),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::GpuAdapter;

    fn gpu(vendor: GpuVendor, name: &str, vram_mb: u64, discrete: bool) -> GpuAdapter {
        GpuAdapter {
            vendor,
            name: name.into(),
            vram_mb,
            index: 0,
            discrete,
            cuda: None,
        }
    }

    fn facts(cores: usize, mem: f64, arch: &str, gpus: Vec<GpuAdapter>) -> Facts {
        let os = if arch == "aarch64" && gpus.iter().any(|g| g.vendor == GpuVendor::Apple) {
            "macos"
        } else {
            "windows"
        };
        Facts::from_hardware(cores, Some(mem), &HardwareInfo::new(os, arch, gpus))
    }

    #[test]
    fn i5_8500_with_uhd_630_is_low() {
        let f = facts(
            6,
            16.0,
            "x86_64",
            vec![gpu(
                GpuVendor::Intel,
                "Intel(R) UHD Graphics 630",
                128,
                false,
            )],
        );
        assert_eq!(f.gpu, GpuClass::Integrated);
        let p = classify(&f);
        assert_eq!(p.class, HardwareClass::Low);
        assert_eq!(p.preprocess_threads, 3);
        assert_eq!(p.preprocess_threads_for(false), 3);
        assert_eq!(p.preprocess_threads_for(true), 2);
        let igpu = p.tuning_for(DeviceClass::IntegratedGpu);
        assert_eq!((igpu.hint, igpu.inflight), (PerfHint::Latency, 2));
        let cpu = p.tuning_for(DeviceClass::Cpu);
        assert_eq!((cpu.hint, cpu.inflight), (PerfHint::Latency, 1));
        assert_eq!(p.model_sizes, &[ModelSize::N, ModelSize::S]);
        assert_eq!(p.inflight, 2);
        assert!(p.reasons.contains(&"6 logical cores".to_string()));
        assert!(
            p.reasons
                .iter()
                .any(|r| r == "integrated GPU (Intel(R) UHD Graphics 630)")
        );
        let line = p.summary(3);
        assert!(
            line.starts_with("low-spec: 6 logical cores, 16 GB memory"),
            "{line}"
        );
        assert!(line.contains("3 preprocess threads, 2 in flight"), "{line}");
    }

    #[test]
    fn small_box_without_gpu_is_low() {
        let p = classify(&facts(4, 8.0, "x86_64", vec![]));
        assert_eq!(p.class, HardwareClass::Low);
        assert_eq!(p.preprocess_threads_for(true), 2);
        assert_eq!(p.tuning_for(DeviceClass::Cpu).inflight, 1);
        assert_eq!(p.inflight, 1);
        let p = classify(&facts(2, 4.0, "x86_64", vec![]));
        assert_eq!(p.preprocess_threads_for(false), 1);
    }

    #[test]
    fn rtx_3060_workstation_is_high() {
        let p = classify(&facts(
            16,
            32.0,
            "x86_64",
            vec![
                gpu(GpuVendor::Intel, "Intel(R) UHD Graphics 770", 0, false),
                gpu(GpuVendor::Nvidia, "NVIDIA GeForce RTX 3060", 12288, true),
            ],
        ));
        assert_eq!(p.class, HardwareClass::High);
        assert_eq!(p.preprocess_threads, 8);
        assert_eq!(p.ort_sessions, 2);
        let cuda = p.tuning_for(DeviceClass::Ort { gpu: true });
        assert_eq!((cuda.inflight, cuda.ort_sessions), (2, 2));
        assert_eq!(
            p.tuning_for(DeviceClass::Ort { gpu: false }).ort_sessions,
            1
        );
        assert_eq!(p.model_sizes, &[ModelSize::M, ModelSize::L, ModelSize::X]);
        assert!(p.reasons.iter().any(|r| r.contains("RTX 3060, 12 GB VRAM")));
    }

    #[test]
    fn arc_a770_is_high_with_four_in_flight() {
        let p = classify(&facts(
            16,
            32.0,
            "x86_64",
            vec![gpu(
                GpuVendor::Intel,
                "Intel(R) Arc(TM) A770 Graphics",
                16384,
                true,
            )],
        ));
        assert_eq!(p.class, HardwareClass::High);
        // ort:directml on the Arc: two sessions (16 GB VRAM).
        assert_eq!(p.ort_sessions, 2);
        let t = p.tuning_for(DeviceClass::DiscreteGpu);
        assert_eq!(
            (t.hint, t.inflight, t.clamp_to_streams),
            (PerfHint::Throughput, 4, true)
        );
        assert_eq!(t.effective_inflight(Some(2)), 2);
        assert_eq!(t.effective_inflight(Some(8)), 4);
        assert_eq!(t.effective_inflight(None), 4);
        assert_eq!(p.primary_device(), DeviceClass::DiscreteGpu);
        assert_eq!(p.inflight, 4);
    }

    #[test]
    fn many_cores_without_gpu_run_cpu_throughput() {
        let p = classify(&facts(16, 64.0, "x86_64", vec![]));
        assert_eq!(p.class, HardwareClass::Low, "no GPU on x86");
        let t = p.tuning_for(DeviceClass::Cpu);
        assert_eq!(t.hint, PerfHint::Throughput);
        assert!(t.clamp_to_streams);
        assert_eq!(t.effective_inflight(Some(4)), 4, "inflight = NUM_STREAMS");
        assert_eq!(t.effective_inflight(Some(32)), MAX_INFLIGHT);
        assert_eq!(p.preprocess_threads_for(true), 2);
        assert!(p.summary(2).contains("up to 8 in flight (NUM_STREAMS)"));
    }

    #[test]
    fn m1_is_high() {
        let p = classify(&facts(
            8,
            16.0,
            "aarch64",
            vec![gpu(GpuVendor::Apple, "Apple M1", 0, false)],
        ));
        assert_eq!(p.facts.gpu, GpuClass::Apple);
        assert_eq!(p.class, HardwareClass::High);
        assert_eq!(p.preprocess_threads, 4);
        assert_eq!(p.ort_sessions, 1);
        // 8 cores: CPU inference stays on LATENCY with one in flight.
        assert_eq!(p.tuning_for(DeviceClass::Cpu), DeviceTuning::SERIAL);
    }

    #[test]
    fn gtx_1050_is_mid() {
        let p = classify(&facts(
            8,
            16.0,
            "x86_64",
            vec![gpu(
                GpuVendor::Nvidia,
                "NVIDIA GeForce GTX 1050",
                2048,
                true,
            )],
        ));
        assert_eq!(p.class, HardwareClass::Mid);
        assert_eq!(p.ort_sessions, 1, "2 GB of VRAM: one copy of the model");
        assert_eq!(p.model_sizes, &[ModelSize::S, ModelSize::M]);
    }

    #[test]
    fn low_class_box_with_1070ti_gets_two_sessions() {
        let p = classify(&facts(
            6,
            16.0,
            "x86_64",
            vec![gpu(
                GpuVendor::Nvidia,
                "NVIDIA GeForce GTX 1070 Ti",
                8192,
                true,
            )],
        ));
        assert_eq!(p.class, HardwareClass::Low);
        assert_eq!(p.ort_sessions, 2);
        let t = p.tuning_for(DeviceClass::Ort { gpu: true });
        assert_eq!((t.inflight, t.ort_sessions), (2, 2));
        assert_eq!(
            p.tuning_for(DeviceClass::Ort { gpu: false }).ort_sessions,
            1
        );
        assert!(p.summary(3).contains("2 in flight"), "{}", p.summary(3));
    }

    #[test]
    fn small_vram_dgpu_keeps_one_session() {
        let p = classify(&facts(
            6,
            16.0,
            "x86_64",
            vec![gpu(GpuVendor::Nvidia, "NVIDIA GeForce GT 1030", 2048, true)],
        ));
        assert_eq!(p.ort_sessions, 1);
        assert_eq!(p.tuning_for(DeviceClass::Ort { gpu: true }).inflight, 1);
    }

    #[test]
    fn dgpu_with_little_memory_keeps_one_session() {
        let p = classify(&facts(
            8,
            4.0,
            "x86_64",
            vec![gpu(
                GpuVendor::Nvidia,
                "NVIDIA GeForce GTX 1650",
                4096,
                true,
            )],
        ));
        assert_eq!(p.ort_sessions, 1);
        // Unknown memory counts as enough.
        let f = Facts::from_hardware(
            8,
            None,
            &HardwareInfo::new(
                "windows",
                "x86_64",
                vec![gpu(
                    GpuVendor::Nvidia,
                    "NVIDIA GeForce GTX 1650",
                    4096,
                    true,
                )],
            ),
        );
        assert_eq!(classify(&f).ort_sessions, 2);
    }

    #[test]
    fn igpu_keeps_one_session() {
        let p = classify(&facts(
            12,
            32.0,
            "x86_64",
            vec![gpu(
                GpuVendor::Intel,
                "Intel(R) UHD Graphics 770",
                4096,
                false,
            )],
        ));
        assert_eq!(p.facts.gpu, GpuClass::Integrated);
        assert_eq!(p.ort_sessions, 1);
    }

    #[test]
    fn unknown_memory_is_never_high() {
        let f = Facts::from_hardware(
            16,
            None,
            &HardwareInfo::new(
                "linux",
                "x86_64",
                vec![gpu(GpuVendor::Nvidia, "RTX 4090", 24576, true)],
            ),
        );
        let p = classify(&f);
        assert_eq!(p.class, HardwareClass::Mid);
        assert!(p.reasons.contains(&"memory unknown".to_string()));
    }

    #[test]
    fn forced_class_wins() {
        let f = facts(6, 16.0, "x86_64", vec![]);
        let p = classify_as(&f, HardwareProfile::High.forced());
        assert_eq!(p.class, HardwareClass::High);
        assert!(p.forced);
        assert!(p.reasons.iter().any(|r| r.contains("detected: low")));
        assert_eq!(p.model_sizes, HIGH_SIZES);
        assert!(p.summary(3).starts_with("high-spec (forced): "));
        assert_eq!("HIGH".parse::<HardwareProfile>(), Ok(HardwareProfile::High));
        assert!("ultra".parse::<HardwareProfile>().is_err());
        let v: HardwareProfile = serde_json::from_str(r#""mid""#).unwrap();
        assert_eq!(v, HardwareProfile::Mid);
    }

    #[test]
    fn explicit_settings_override_the_profile() {
        let p = classify(&facts(16, 64.0, "x86_64", vec![]));
        let cpu = p.tuning_with(
            DeviceClass::Cpu,
            Overrides {
                inflight: 2,
                ort_sessions: 0,
            },
        );
        assert_eq!(cpu.inflight, 2);
        assert!(!cpu.clamp_to_streams, "explicit counts are not clamped");
        assert_eq!(cpu.hint, PerfHint::Throughput);
        assert_eq!(cpu.effective_inflight(Some(1)), 2);
        let ort = p.tuning_with(
            DeviceClass::Ort { gpu: false },
            Overrides {
                inflight: 0,
                ort_sessions: 3,
            },
        );
        assert_eq!((ort.inflight, ort.ort_sessions), (3, 3));
        let ort = p.tuning_with(
            DeviceClass::Ort { gpu: true },
            Overrides {
                inflight: 1,
                ort_sessions: 3,
            },
        );
        assert_eq!((ort.inflight, ort.ort_sessions), (1, 1));
        let big = p.tuning_with(
            DeviceClass::IntegratedGpu,
            Overrides {
                inflight: 100,
                ort_sessions: 0,
            },
        );
        assert_eq!(big.inflight, MAX_INFLIGHT);
    }

    #[test]
    fn model_sizes_from_names() {
        let cases = [
            ("yolov8n", Some(ModelSize::N)),
            ("yolov9t", Some(ModelSize::N)),
            ("yolov9c", Some(ModelSize::L)),
            ("yolov9e", Some(ModelSize::X)),
            ("yolov10b", Some(ModelSize::M)),
            ("yolo11x", Some(ModelSize::X)),
            ("yolo26s.onnx", Some(ModelSize::S)),
            ("YOLO12M", Some(ModelSize::M)),
            ("rt-detrv2-s", Some(ModelSize::S)),
            ("rt-detrv2-ms", Some(ModelSize::M)),
            ("rt-detrv2-x", Some(ModelSize::X)),
            ("dfine-l", Some(ModelSize::L)),
            ("rfdetr-nano", Some(ModelSize::N)),
            ("rfdetr-small", Some(ModelSize::S)),
            ("rfdetr-base", Some(ModelSize::M)),
            ("rfdetr-medium", Some(ModelSize::M)),
            ("rfdetr-large", Some(ModelSize::L)),
            ("deimv2-atto", Some(ModelSize::N)),
            ("deimv2-pico", Some(ModelSize::N)),
            ("deimv2-x", Some(ModelSize::X)),
            ("IPcam-general", None),
            ("custom", None),
            ("yolo", None),
        ];
        for (name, want) in cases {
            assert_eq!(model_size(name, false), want, "{name}");
        }
        // The IPcam YOLOv5 models (and delivery / package) count as `s`.
        for name in ["IPcam-general", "IPcam-dark", "delivery", "package"] {
            assert_eq!(model_size(name, true), Some(ModelSize::S), "{name}");
        }
    }

    #[test]
    fn every_catalog_model_has_a_size() {
        use crate::model::ModelFamilyKind;
        use crate::resources::catalog::Provides;
        let all = crate::resources::catalog::MODELS
            .iter()
            .chain(crate::resources::catalog::EXPORT_MODELS.iter());
        for r in all {
            if let Provides::Model { name, family } = r.provides {
                assert!(
                    model_size(name, family == ModelFamilyKind::Yolo5).is_some(),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn facts_pick_the_biggest_gpu() {
        let f = facts(
            8,
            32.0,
            "x86_64",
            vec![
                gpu(GpuVendor::Intel, "UHD", 0, false),
                gpu(GpuVendor::Nvidia, "small", 2048, true),
                gpu(GpuVendor::Nvidia, "big", 8192, true),
            ],
        );
        assert_eq!(f.gpu, GpuClass::Discrete { vram_mb: 8192 });
        assert_eq!(f.gpu_name.as_deref(), Some("big"));
        let detected = Facts::detect();
        assert!(detected.logical_cores >= 1);
    }
}
