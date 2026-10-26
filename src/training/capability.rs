//! Can this machine train, on what, and which families and sizes fit ([`capability`], pure over
//! [`HardwareInfo`]). Training needs one of:
//!
//! - an NVIDIA GPU (Windows / Linux x86_64) with compute capability 6.1 or newer that the pinned
//!   PyTorch build has kernels for, and a driver new enough for it: the CUDA 12.6 build
//!   ([`TrainLane::Cu126`], kernels for 5.0-9.0, any CUDA 12 driver) for Pascal to Hopper, the
//!   CUDA 13.0 build ([`TrainLane::Cu130`], 7.5-12.0, a CUDA 13 driver) for Blackwell;
//! - Apple silicon (macOS arm64) with MPS ([`TrainLane::Mps`]): the families whose 1-epoch smoke
//!   run passed there ([`TrainFamily::mps_ok`]).
//!
//! Every qualifying NVIDIA GPU is a [`Capability::cuda_options`] entry (the one with the most VRAM
//! first: the automatic pick); a run may name another one (`gpu: cuda:N`, [`Capability::for_gpu`]),
//! which then sets up that GPU's lane. The ordinal is the CUDA driver's, which is PyTorch's as long
//! as the child sees the service's `CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER` (it does: the
//! export runner passes both through).
//!
//! Otherwise the answer says why and points at the by-hand path (`scripts/train_model.py` in an
//! environment from `scripts/requirements-model-train.txt`, on another machine). Sizes whose
//! estimated VRAM ([`super::estimate`]) does not fit even at batch 1 are marked as not fitting.

use super::estimate::{self, Target};
use crate::backend::detect::{ComputeCapability, GpuVendor, HardwareInfo};
use crate::resources::catalog::TrainFamily;
use crate::resources::export::train::TrainLane;
use anyhow::{Result, bail};
use serde::Serialize;

/// Oldest compute capability offered (Pascal consumer cards, GTX 10xx).
pub const MIN_CC: ComputeCapability = ComputeCapability::new(6, 1);
/// Kernels of the PyTorch 2.14 CUDA 12.6 wheels on Windows (`TORCH_CUDA_ARCH_LIST` of the build).
pub const CU126_WINDOWS: &[(u32, u32)] = &[
    (5, 0),
    (6, 0),
    (6, 1),
    (7, 0),
    (7, 5),
    (8, 0),
    (8, 6),
    (9, 0),
];
/// ... on Linux.
pub const CU126_LINUX: &[(u32, u32)] = &[(5, 0), (6, 0), (7, 0), (7, 5), (8, 0), (8, 6), (9, 0)];
/// Kernels of the CUDA 13.0 wheels (Windows and Linux).
pub const CU130: &[(u32, u32)] = &[(7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0)];
/// Oldest driver CUDA version of each lane.
pub const CU126_MIN_DRIVER: (u32, u32) = (12, 0);
pub const CU130_MIN_DRIVER: (u32, u32) = (13, 0);
/// Share of an Apple machine's unified memory the estimate may use.
pub const MPS_MEMORY_SHARE: f64 = 0.6;

/// A cubin of `sm_X.Y` runs on every GPU of major X and minor >= Y.
pub fn has_kernels(table: &[(u32, u32)], cc: ComputeCapability) -> bool {
    table
        .iter()
        .any(|&(ma, mi)| ma == cc.major && mi <= cc.minor)
}

/// The device a run trains on.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Accelerator {
    Cuda {
        /// CUDA device ordinal (`--device cuda:N`).
        ordinal: u32,
        name: String,
        compute_capability: ComputeCapability,
        vram_mb: u64,
        lane: TrainLane,
    },
    Mps {
        name: String,
        /// Unified memory the estimates may use, MB.
        memory_mb: u64,
    },
}

impl Accelerator {
    pub fn lane(&self) -> TrainLane {
        match self {
            Accelerator::Cuda { lane, .. } => *lane,
            Accelerator::Mps { .. } => TrainLane::Mps,
        }
    }

    /// The GPU's name (`NVIDIA GeForce RTX 3090`, `Apple M1`).
    pub fn name(&self) -> &str {
        match self {
            Accelerator::Cuda { name, .. } | Accelerator::Mps { name, .. } => name,
        }
    }

    /// Memory the estimates compare against, MB (0 = unknown).
    pub fn memory_mb(&self) -> u64 {
        match self {
            Accelerator::Cuda { vram_mb, .. } => *vram_mb,
            Accelerator::Mps { memory_mb, .. } => *memory_mb,
        }
    }

    /// `--device` of the training script.
    pub fn device_arg(&self) -> String {
        match self {
            Accelerator::Cuda { ordinal, .. } => format!("cuda:{ordinal}"),
            Accelerator::Mps { .. } => "mps".to_string(),
        }
    }

    pub fn target(&self) -> Target {
        match self {
            Accelerator::Cuda {
                compute_capability, ..
            } => Target::Cuda(*compute_capability),
            Accelerator::Mps { .. } => Target::Mps,
        }
    }

    /// The PyTorch backend the scripts run on, for the page: `Metal (Apple M1)`,
    /// `CUDA cuda:0 (NVIDIA GeForce RTX 3060)`.
    pub fn torch_backend(&self) -> String {
        match self {
            Accelerator::Cuda { name, .. } => format!("CUDA {} ({name})", self.device_arg()),
            Accelerator::Mps { name, .. } => format!("Metal ({name})"),
        }
    }

    /// One GPU choice of the page and the errors: `NVIDIA GeForce RTX 3090 (cuda:1, 24 GB,
    /// PyTorch cu126)`, `Apple M1 (mps)`.
    pub fn choice_label(&self) -> String {
        match self {
            Accelerator::Cuda {
                name,
                vram_mb,
                lane,
                ..
            } => format!(
                "{name} ({}, {} GB, PyTorch {})",
                self.device_arg(),
                vram_mb / 1024,
                lane.as_str()
            ),
            Accelerator::Mps { name, .. } => format!("{name} (mps)"),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Accelerator::Cuda {
                name,
                compute_capability,
                vram_mb,
                lane,
                ..
            } => format!(
                "{name} (compute capability {compute_capability}, {} GB, PyTorch {})",
                vram_mb / 1024,
                lane.as_str()
            ),
            Accelerator::Mps { name, memory_mb } => {
                format!(
                    "{name} (Apple MPS, {} GB unified memory usable)",
                    memory_mb / 1024
                )
            }
        }
    }
}

/// One size of a family on this machine.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SizeFit {
    pub size: &'static str,
    /// Trainable here (family available and the estimate fits at some batch).
    pub fits: bool,
    pub default_imgsz: u32,
    /// Largest batch whose estimate fits (None = does not fit / unknown memory).
    pub suggested_batch: Option<u32>,
    /// Estimate at the suggested batch (or 8), per 100 training images and epoch.
    pub vram_mb: Option<u64>,
    pub epoch_s_per_100: Option<f64>,
    pub latency: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

/// One family on this machine.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FamilyFit {
    pub family: TrainFamily,
    pub title: &'static str,
    pub license: &'static str,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    pub sizes: Vec<SizeFit>,
}

/// What this machine can train.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Capability {
    pub available: bool,
    /// The automatic pick (the first of [`Self::cuda_options`] on a PC with NVIDIA GPUs).
    pub accelerator: Option<Accelerator>,
    /// Every NVIDIA GPU PyTorch can run on here, the automatic pick first (most VRAM first; the
    /// driver's order among equals). Empty on Apple silicon and without a usable NVIDIA GPU.
    pub cuda_options: Vec<Accelerator>,
    /// NVIDIA GPUs the CUDA driver numbers but PyTorch cannot run on: (ordinal, why).
    #[serde(skip)]
    passed_over: Vec<(u32, String)>,
    /// Why training is not available, or notes (GPUs passed over).
    pub reasons: Vec<String>,
    pub families: Vec<FamilyFit>,
}

impl Capability {
    pub fn family(&self, f: TrainFamily) -> Option<&FamilyFit> {
        self.families.iter().find(|x| x.family == f)
    }

    /// Whether the page offers a GPU choice (more than one usable NVIDIA GPU).
    pub fn has_gpu_choice(&self) -> bool {
        self.cuda_options.len() > 1
    }

    /// The accelerator a run with `gpu` uses ([`gpu_arg`]: `cuda:N` of [`Self::cuda_options`],
    /// `mps` on Apple silicon; None = the automatic pick). Ok(None): no GPU and none asked for
    /// (the caller says what that means for it). An unknown, unusable or foreign choice is an
    /// error naming the usable ones.
    pub fn resolve_gpu(&self, gpu: Option<&str>) -> Result<Option<Accelerator>> {
        let Some(want) = gpu_arg(gpu)? else {
            return Ok(self.accelerator.clone());
        };
        let Some(auto) = &self.accelerator else {
            bail!("GPU {want} was chosen, but {}", self.summary());
        };
        if let Accelerator::Mps { .. } = auto {
            if want == "mps" {
                return Ok(Some(auto.clone()));
            }
            bail!(
                "GPU {want}: this machine runs PyTorch on {} only; there is no GPU to choose \
                 (leave it automatic or use mps)",
                auto.torch_backend()
            );
        }
        if let Some(a) = self.cuda_options.iter().find(|a| a.device_arg() == want) {
            return Ok(Some(a.clone()));
        }
        let usable = self
            .cuda_options
            .iter()
            .map(Accelerator::choice_label)
            .collect::<Vec<_>>()
            .join("; ");
        let why = want
            .strip_prefix("cuda:")
            .and_then(|n| n.parse::<u32>().ok())
            .and_then(|n| self.passed_over.iter().find(|(o, _)| *o == n))
            .map(|(_, why)| format!("PyTorch cannot train on it here ({why})"));
        match why {
            Some(why) => bail!("GPU {want}: {why}; usable: {usable}"),
            None if want == "mps" => {
                bail!("GPU mps: Apple MPS is on Apple silicon only; usable here: {usable}")
            }
            None => bail!("GPU {want}: no such CUDA device here; usable: {usable}"),
        }
    }

    /// This capability on the GPU `gpu` names ([`Self::resolve_gpu`]): its accelerator, the
    /// families' fits and suggested batches re-estimated for that GPU's memory.
    pub fn for_gpu(&self, gpu: Option<&str>) -> Result<Capability> {
        let accel = self.resolve_gpu(gpu)?;
        if accel == self.accelerator {
            return Ok(self.clone());
        }
        Ok(Capability {
            families: families(accel.as_ref()),
            accelerator: accel,
            ..self.clone()
        })
    }

    /// One line for logs and the CLI.
    pub fn summary(&self) -> String {
        match &self.accelerator {
            Some(a) => format!("training available on {}", a.describe()),
            None => format!(
                "training is not available on this machine: {}",
                self.reasons.join("; ")
            ),
        }
    }
}

/// The by-hand path, appended to "not available" answers.
pub const BY_HAND: &str = "you can still train on another machine with scripts/train_model.py \
    (Python 3.11, packages of scripts/requirements-model-train.txt) and copy the .onnx and .yaml \
    into the models folder (Add a model by path)";

/// A GPU choice as typed (`cuda:1`, ` CUDA:1 `; empty or `auto` = the automatic pick) in its
/// canonical form (`cuda:N`, `mps`); anything else is an error saying what is accepted.
pub fn gpu_arg(gpu: Option<&str>) -> Result<Option<String>> {
    let Some(g) = gpu.map(|g| g.trim().to_ascii_lowercase()) else {
        return Ok(None);
    };
    if g.is_empty() || g == "auto" {
        return Ok(None);
    }
    if g == "mps" {
        return Ok(Some(g));
    }
    match g.strip_prefix("cuda:").map(|n| n.trim().parse::<u32>()) {
        Some(Ok(n)) => Ok(Some(format!("cuda:{n}"))),
        _ => bail!(
            "GPU '{g}' is not cuda:N (N = the CUDA device number, see `train capability`), mps or \
             auto"
        ),
    }
}

/// Decide what `hw` can train. `ram_gb`: total system memory (Apple unified memory), if known.
pub fn capability(hw: &HardwareInfo, ram_gb: Option<f64>) -> Capability {
    let mut reasons = Vec::new();
    let mut passed_over = Vec::new();
    let mut cuda_options = Vec::new();
    let accel = if hw.is_apple_silicon() {
        let name = hw
            .gpus_of(GpuVendor::Apple)
            .next()
            .map_or_else(|| "Apple GPU".to_string(), |g| g.name.clone());
        let memory_mb = ram_gb.map_or(0, |g| estimate::to_u64(g * 1024.0 * MPS_MEMORY_SHARE));
        Some(Accelerator::Mps { name, memory_mb })
    } else {
        cuda_options = cuda_gpus(hw, &mut reasons, &mut passed_over);
        cuda_options.first().cloned()
    };
    if accel.is_none() && reasons.is_empty() {
        reasons.push(if hw.is_macos() {
            "Intel Macs have no PyTorch 2.14 GPU support (Apple MPS needs Apple silicon)"
                .to_string()
        } else if !hw.has_vendor(GpuVendor::Nvidia) {
            format!(
                "no NVIDIA GPU found (training needs CUDA on {}/{} or Apple silicon; Intel and \
                 AMD GPUs are not supported by the pinned PyTorch)",
                hw.os, hw.arch
            )
        } else {
            "no usable NVIDIA GPU".to_string()
        });
    }
    if accel.is_none() {
        reasons.push(BY_HAND.to_string());
    }
    Capability {
        available: accel.is_some(),
        families: families(accel.as_ref()),
        accelerator: accel,
        cuda_options,
        passed_over,
        reasons,
    }
}

fn families(accel: Option<&Accelerator>) -> Vec<FamilyFit> {
    TrainFamily::ALL
        .into_iter()
        .map(|f| family_fit(f, accel))
        .collect()
}

/// The NVIDIA GPUs PyTorch can train on, the one with the most VRAM first (a stable order: the
/// driver's among equals). Every GPU passed over adds its reason (and, when the driver numbers
/// it, its ordinal to `passed_over`).
fn cuda_gpus(
    hw: &HardwareInfo,
    reasons: &mut Vec<String>,
    passed_over: &mut Vec<(u32, String)>,
) -> Vec<Accelerator> {
    let nvidia: Vec<_> = hw.gpus_of(GpuVendor::Nvidia).collect();
    if nvidia.is_empty() {
        return Vec::new();
    }
    let platform_ok = matches!(
        (hw.os.as_str(), hw.arch.as_str()),
        ("windows", "x86_64") | ("linux", "x86_64")
    );
    if !platform_ok {
        reasons.push(format!(
            "no pinned PyTorch CUDA build for {}/{} (Windows and Linux x86_64 only)",
            hw.os, hw.arch
        ));
        return Vec::new();
    }
    let Some(driver) = hw.cuda_driver else {
        reasons.push(
            "the NVIDIA driver does not report a CUDA version (driver missing or too old); \
             install a current NVIDIA driver (CUDA 12 or newer)"
                .to_string(),
        );
        return Vec::new();
    };
    let cu126 = if hw.is_windows() {
        CU126_WINDOWS
    } else {
        CU126_LINUX
    };
    let mut found: Vec<Accelerator> = Vec::new();
    for g in nvidia {
        let Some(cuda) = g.cuda else {
            reasons.push(format!("{}: not visible to the CUDA driver", g.name));
            continue;
        };
        let cc = cuda.compute_capability;
        let lane = if cc < MIN_CC {
            Err(format!(
                "compute capability {cc} is older than {MIN_CC} (Pascal GTX 10xx)"
            ))
        } else if has_kernels(cu126, cc) {
            if (driver.major, driver.minor) < CU126_MIN_DRIVER {
                Err(format!(
                    "the driver supports CUDA {driver}; PyTorch's CUDA 12.6 build needs a CUDA 12 \
                     driver: update the NVIDIA driver"
                ))
            } else {
                Ok(TrainLane::Cu126)
            }
        } else if has_kernels(CU130, cc) {
            if (driver.major, driver.minor) < CU130_MIN_DRIVER {
                Err(format!(
                    "compute capability {cc} needs PyTorch's CUDA 13.0 build, which needs a CUDA \
                     13 driver (this one supports {driver}): update the NVIDIA driver"
                ))
            } else {
                Ok(TrainLane::Cu130)
            }
        } else {
            Err(format!(
                "the pinned PyTorch 2.14 builds have no kernels for compute capability {cc}"
            ))
        };
        match lane {
            Ok(lane) => found.push(Accelerator::Cuda {
                ordinal: cuda.ordinal,
                name: g.name.clone(),
                compute_capability: cc,
                vram_mb: g.vram_mb,
                lane,
            }),
            Err(why) => {
                reasons.push(format!("{}: {why}", g.name));
                passed_over.push((cuda.ordinal, format!("{}: {why}", g.name)));
            }
        }
    }
    // Stable: equal VRAM keeps the driver's order (the first of them is the automatic pick).
    found.sort_by_key(|a| std::cmp::Reverse(a.memory_mb()));
    found
}

fn family_fit(f: TrainFamily, accel: Option<&Accelerator>) -> FamilyFit {
    let why = match accel {
        None => Some("no GPU to train on".to_string()),
        Some(Accelerator::Mps { .. }) if !f.mps_ok() => f.mps_note().map(str::to_string),
        _ => None,
    };
    let available = why.is_none();
    let sizes = f
        .sizes()
        .iter()
        .map(|&size| {
            let imgsz = f.default_imgsz(size);
            let mem = accel.map_or(0, Accelerator::memory_mb);
            let batch = if mem > 0 {
                estimate::suggest_batch(f, size, imgsz, mem)
            } else {
                Some(8)
            };
            let est = accel.and_then(|a| {
                estimate::estimate(f, size, imgsz, batch.unwrap_or(1), 100, 1, a.target())
            });
            let (fits, why) = if !available {
                (false, None)
            } else if batch.is_none() {
                (
                    false,
                    Some(format!(
                        "needs more than the {} GB of this GPU even at batch 1 (estimate)",
                        mem / 1024
                    )),
                )
            } else {
                (true, None)
            };
            SizeFit {
                size,
                fits,
                default_imgsz: imgsz,
                suggested_batch: batch.filter(|_| available),
                vram_mb: est.map(|e| e.vram_mb),
                epoch_s_per_100: est.map(|e| e.epoch_s),
                latency: estimate::latency_hint(f, size),
                why,
            }
        })
        .collect();
    FamilyFit {
        family: f,
        title: f.title(),
        license: f.license(),
        available,
        why,
        sizes,
    }
}

/// A capability for the machine the tests run on, so jobs planned with it find a training
/// environment for this platform: the M1 on macOS arm64, else an RTX 3060 (8.6, 12 GB, CUDA 12.8
/// driver) on this OS.
#[cfg(test)]
pub(crate) fn host_for_tests() -> Capability {
    use crate::backend::detect::{CudaInfo, GpuAdapter, apple_gpu};
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let hw = if os == "macos" && arch == "aarch64" {
        HardwareInfo::new(os, arch, vec![apple_gpu("Apple M1")])
    } else {
        HardwareInfo::new(
            os,
            arch,
            vec![GpuAdapter {
                vendor: GpuVendor::Nvidia,
                name: "NVIDIA GeForce RTX 3060".into(),
                vram_mb: 12288,
                index: 0,
                discrete: true,
                cuda: Some(CudaInfo {
                    ordinal: 0,
                    compute_capability: ComputeCapability::new(8, 6),
                }),
            }],
        )
        .with_cuda_driver(12, 8)
    };
    capability(&hw, Some(16.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::{CudaInfo, GpuAdapter, apple_gpu};

    fn nvidia(name: &str, cc: (u32, u32), vram_mb: u64, ordinal: u32) -> GpuAdapter {
        GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: name.into(),
            vram_mb,
            index: ordinal,
            discrete: true,
            cuda: Some(CudaInfo {
                ordinal,
                compute_capability: ComputeCapability::new(cc.0, cc.1),
            }),
        }
    }

    fn intel_igpu() -> GpuAdapter {
        GpuAdapter {
            vendor: GpuVendor::Intel,
            name: "Intel(R) UHD Graphics 630".into(),
            vram_mb: 0,
            index: 0,
            discrete: false,
            cuda: None,
        }
    }

    fn lane(c: &Capability) -> Option<TrainLane> {
        c.accelerator.as_ref().map(Accelerator::lane)
    }

    #[test]
    fn kernel_tables_follow_major_compatibility() {
        let cc = ComputeCapability::new;
        assert!(has_kernels(CU126_LINUX, cc(6, 1)), "sm_60 runs on 6.1");
        assert!(has_kernels(CU126_LINUX, cc(8, 9)), "sm_86 runs on Ada");
        assert!(!has_kernels(CU126_LINUX, cc(12, 0)));
        assert!(has_kernels(CU130, cc(12, 0)));
        assert!(!has_kernels(CU130, cc(6, 1)));
        assert!(!has_kernels(CU130, cc(11, 0)));
    }

    #[test]
    fn pascal_on_windows_uses_the_cuda_12_build() {
        let hw = HardwareInfo::new(
            "windows",
            "x86_64",
            vec![
                intel_igpu(),
                nvidia("NVIDIA GeForce GTX 1070", (6, 1), 8192, 0),
            ],
        )
        .with_cuda_driver(12, 8);
        let c = capability(&hw, Some(16.0));
        assert!(c.available, "{c:?}");
        assert_eq!(lane(&c), Some(TrainLane::Cu126));
        assert_eq!(c.accelerator.as_ref().unwrap().device_arg(), "cuda:0");
        let t = c.accelerator.as_ref().unwrap().torch_backend();
        assert!(t.starts_with("CUDA cuda:0 ("), "{t}");
        for f in &c.families {
            assert!(f.available, "{f:?}");
        }
        // 8 GB: YOLO26 n..l fit, RF-DETR large too at a small batch.
        let y = c.family(TrainFamily::Yolo26).unwrap();
        assert!(y.sizes.iter().all(|s| s.fits), "{y:?}");
        assert_eq!(y.sizes[1].suggested_batch, Some(16));
    }

    #[test]
    fn blackwell_needs_a_cuda_13_driver() {
        let gpu = || vec![nvidia("NVIDIA GeForce RTX 5070", (12, 0), 12288, 0)];
        let old = capability(
            &HardwareInfo::new("linux", "x86_64", gpu()).with_cuda_driver(12, 8),
            None,
        );
        assert!(!old.available);
        assert!(
            old.reasons.iter().any(|r| r.contains("CUDA 13 driver")),
            "{:?}",
            old.reasons
        );
        assert!(old.reasons.iter().any(|r| r.contains("train_model.py")));
        let new = capability(
            &HardwareInfo::new("linux", "x86_64", gpu()).with_cuda_driver(13, 0),
            None,
        );
        assert_eq!(lane(&new), Some(TrainLane::Cu130));
    }

    #[test]
    fn old_or_unknown_gpus_explain_why() {
        let maxwell = HardwareInfo::new(
            "windows",
            "x86_64",
            vec![nvidia("GTX 960", (5, 2), 4096, 0)],
        )
        .with_cuda_driver(12, 4);
        let c = capability(&maxwell, None);
        assert!(!c.available);
        assert!(c.reasons[0].contains("older than 6.1"), "{:?}", c.reasons);
        let no_driver = HardwareInfo::new(
            "linux",
            "x86_64",
            vec![nvidia("RTX 3060", (8, 6), 12288, 0)],
        );
        assert!(capability(&no_driver, None).reasons[0].contains("driver"));
        let old_driver = HardwareInfo::new(
            "linux",
            "x86_64",
            vec![nvidia("RTX 3060", (8, 6), 12288, 0)],
        )
        .with_cuda_driver(11, 8);
        assert!(capability(&old_driver, None).reasons[0].contains("CUDA 12 driver"));
        let intel = HardwareInfo::new("windows", "x86_64", vec![intel_igpu()]);
        let c = capability(&intel, Some(16.0));
        assert!(!c.available);
        assert!(c.reasons[0].contains("no NVIDIA GPU"), "{:?}", c.reasons);
        assert!(
            c.families
                .iter()
                .all(|f| !f.available && f.sizes.iter().all(|s| !s.fits))
        );
        let arm = HardwareInfo::new("linux", "aarch64", vec![nvidia("Orin", (8, 7), 8192, 0)])
            .with_cuda_driver(12, 6);
        assert!(capability(&arm, None).reasons[0].contains("x86_64 only"));
    }

    #[test]
    fn the_largest_qualifying_gpu_wins() {
        let hw = HardwareInfo::new(
            "linux",
            "x86_64",
            vec![
                nvidia("GTX 1060", (6, 1), 6144, 0),
                nvidia("RTX 3090", (8, 6), 24576, 1),
                nvidia("GTX 750", (5, 0), 2048, 2),
            ],
        )
        .with_cuda_driver(12, 6);
        let c = capability(&hw, None);
        match c.accelerator.as_ref().unwrap() {
            Accelerator::Cuda { ordinal, name, .. } => {
                assert_eq!((*ordinal, name.as_str()), (1, "RTX 3090"));
            }
            other => panic!("{other:?}"),
        }
        assert!(c.reasons.iter().any(|r| r.contains("GTX 750")));
    }

    /// A GTX 1060 (6 GB, cu126), an RTX 5090 (32 GB, Blackwell: cu130), an RTX 3090 (24 GB,
    /// cu126) and a Maxwell card PyTorch cannot run on, on a CUDA 13 driver.
    fn three_gpus() -> HardwareInfo {
        HardwareInfo::new(
            "windows",
            "x86_64",
            vec![
                intel_igpu(),
                nvidia("NVIDIA GeForce GTX 1060", (6, 1), 6144, 0),
                nvidia("NVIDIA GeForce RTX 5090", (12, 0), 32768, 1),
                nvidia("NVIDIA GeForce GTX 750", (5, 0), 2048, 2),
                nvidia("NVIDIA GeForce RTX 3090", (8, 6), 24576, 3),
            ],
        )
        .with_cuda_driver(13, 0)
    }

    fn ordinals(v: &[Accelerator]) -> Vec<(String, TrainLane)> {
        v.iter().map(|a| (a.device_arg(), a.lane())).collect()
    }

    #[test]
    fn every_qualifying_gpu_is_an_option_most_vram_first() {
        let c = capability(&three_gpus(), Some(32.0));
        assert_eq!(
            ordinals(&c.cuda_options),
            [
                ("cuda:1".to_string(), TrainLane::Cu130),
                ("cuda:3".to_string(), TrainLane::Cu126),
                ("cuda:0".to_string(), TrainLane::Cu126),
            ]
        );
        assert!(c.has_gpu_choice());
        // The automatic pick is unchanged: the first option, the most VRAM.
        assert_eq!(c.accelerator.as_ref(), c.cuda_options.first());
        assert_eq!(c.accelerator.as_ref().unwrap().device_arg(), "cuda:1");
        // Only the unqualified card is a reason (no by-hand note: training is available).
        assert_eq!(c.reasons.len(), 1, "{:?}", c.reasons);
        assert!(
            c.reasons[0].starts_with("NVIDIA GeForce GTX 750: compute capability 5.0"),
            "{:?}",
            c.reasons
        );
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["cuda_options"].as_array().unwrap().len(), 3);
        assert_eq!(v["cuda_options"][1]["ordinal"], 3);
        assert_eq!(v["cuda_options"][1]["lane"], "cu126");
        assert!(v.get("passed_over").is_none());
        assert_eq!(
            c.cuda_options[1].choice_label(),
            "NVIDIA GeForce RTX 3090 (cuda:3, 24 GB, PyTorch cu126)"
        );
        // On a CUDA 12 driver the Blackwell card drops out with its reason; equal VRAM keeps
        // the driver's order.
        let mut hw = three_gpus().with_cuda_driver(12, 8);
        hw.gpus[4].vram_mb = 6144;
        let c = capability(&hw, None);
        assert_eq!(
            ordinals(&c.cuda_options),
            [
                ("cuda:0".to_string(), TrainLane::Cu126),
                ("cuda:3".to_string(), TrainLane::Cu126),
            ]
        );
        assert!(
            c.reasons
                .iter()
                .any(|r| r.contains("RTX 5090") && r.contains("CUDA 13 driver"))
        );
        // One GPU, Apple silicon, none: nothing to choose.
        let one = HardwareInfo::new(
            "linux",
            "x86_64",
            vec![nvidia("RTX 3060", (8, 6), 12288, 0)],
        )
        .with_cuda_driver(12, 6);
        let c = capability(&one, None);
        assert_eq!(c.cuda_options.len(), 1);
        assert!(!c.has_gpu_choice());
        let mac = capability(
            &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
            Some(16.0),
        );
        assert!(mac.cuda_options.is_empty() && !mac.has_gpu_choice());
        let none = capability(&HardwareInfo::new("linux", "x86_64", vec![]), None);
        assert!(none.cuda_options.is_empty());
    }

    #[test]
    fn a_chosen_gpu_resolves_to_its_accelerator_or_says_why_not() {
        let c = capability(&three_gpus(), Some(32.0));
        // Automatic.
        for g in [None, Some(""), Some(" auto ")] {
            assert_eq!(c.resolve_gpu(g).unwrap(), c.accelerator, "{g:?}");
        }
        // A usable one, as typed.
        let a = c.resolve_gpu(Some(" CUDA:3 ")).unwrap().unwrap();
        assert_eq!(
            (a.device_arg(), a.lane()),
            ("cuda:3".into(), TrainLane::Cu126)
        );
        // Its capability: the families re-estimated for 6 GB instead of 32 GB.
        let small = c.for_gpu(Some("cuda:0")).unwrap();
        assert_eq!(small.accelerator.as_ref().unwrap().device_arg(), "cuda:0");
        assert_eq!(small.cuda_options, c.cuda_options);
        let batch = |c: &Capability| {
            c.family(TrainFamily::Rfdetr)
                .unwrap()
                .sizes
                .iter()
                .map(|s| s.suggested_batch)
                .collect::<Vec<_>>()
        };
        assert_ne!(batch(&small), batch(&c));
        assert_eq!(c.for_gpu(None).unwrap(), c);
        // Passed over: the reason and the usable ones.
        let e = c.resolve_gpu(Some("cuda:2")).unwrap_err().to_string();
        assert!(e.contains("GTX 750") && e.contains("older than 6.1"), "{e}");
        assert!(e.contains("usable: NVIDIA GeForce RTX 5090 (cuda:1"), "{e}");
        assert!(e.contains("cuda:3") && e.contains("cuda:0"), "{e}");
        // Unknown and malformed.
        let e = c.resolve_gpu(Some("cuda:7")).unwrap_err().to_string();
        assert!(
            e.contains("no such CUDA device") && e.contains("cuda:1"),
            "{e}"
        );
        for bad in ["gpu1", "cuda", "cuda:x", "1"] {
            let e = c.resolve_gpu(Some(bad)).unwrap_err().to_string();
            assert!(e.contains("is not cuda:N"), "{bad}: {e}");
        }
        let e = c.resolve_gpu(Some("mps")).unwrap_err().to_string();
        assert!(e.contains("Apple silicon only"), "{e}");
        // A Mac: mps or automatic only.
        let mac = capability(
            &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
            Some(16.0),
        );
        assert_eq!(mac.resolve_gpu(Some("MPS")).unwrap(), mac.accelerator);
        let e = mac.resolve_gpu(Some("cuda:0")).unwrap_err().to_string();
        assert!(
            e.contains("Metal (Apple M1") && e.contains("no GPU to choose"),
            "{e}"
        );
        // No GPU: a choice names why.
        let none = capability(&HardwareInfo::new("linux", "x86_64", vec![]), None);
        assert_eq!(none.resolve_gpu(None).unwrap(), None);
        let e = none.resolve_gpu(Some("cuda:0")).unwrap_err().to_string();
        assert!(e.contains("not available"), "{e}");
    }

    #[test]
    fn small_cards_hide_the_sizes_that_cannot_fit() {
        let hw = HardwareInfo::new(
            "windows",
            "x86_64",
            vec![nvidia("GTX 1050", (6, 1), 2048, 0)],
        )
        .with_cuda_driver(12, 4);
        let c = capability(&hw, None);
        let rf = c.family(TrainFamily::Rfdetr).unwrap();
        let large = rf.sizes.iter().find(|s| s.size == "large").unwrap();
        assert!(!large.fits);
        assert!(large.why.as_deref().unwrap().contains("batch 1"));
        let yolo = c.family(TrainFamily::Yolo26).unwrap();
        assert!(yolo.sizes[0].fits);
    }

    #[test]
    fn apple_silicon_trains_what_passed_the_mps_smoke_test() {
        let hw = HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]);
        let c = capability(&hw, Some(16.0));
        assert!(c.available);
        assert_eq!(lane(&c), Some(TrainLane::Mps));
        assert_eq!(c.accelerator.as_ref().unwrap().device_arg(), "mps");
        let t = c.accelerator.as_ref().unwrap().torch_backend();
        assert!(t.starts_with("Metal ("), "{t}");
        assert!(c.family(TrainFamily::Yolo26).unwrap().available);
        assert!(c.family(TrainFamily::Rfdetr).unwrap().available);
        assert!(c.family(TrainFamily::Rtdetrv2).unwrap().available);
        let d = c.family(TrainFamily::Dfine).unwrap();
        assert!(!d.available);
        assert!(d.why.as_deref().unwrap().contains("MPS"));
        assert!(c.summary().contains("Apple MPS"));
        let intel_mac = HardwareInfo::new("macos", "x86_64", vec![]);
        assert!(capability(&intel_mac, None).reasons[0].contains("Intel Macs"));
    }
}
