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
//! Otherwise the answer says why and points at the by-hand path (`scripts/train_model.py` in an
//! environment from `scripts/requirements-model-train.txt`, on another machine). Sizes whose
//! estimated VRAM ([`super::estimate`]) does not fit even at batch 1 are marked as not fitting.

use super::estimate::{self, Target};
use crate::backend::detect::{ComputeCapability, GpuVendor, HardwareInfo};
use crate::resources::catalog::TrainFamily;
use crate::resources::export::train::TrainLane;
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
    pub accelerator: Option<Accelerator>,
    /// Why training is not available, or notes (GPUs passed over).
    pub reasons: Vec<String>,
    pub families: Vec<FamilyFit>,
}

impl Capability {
    pub fn family(&self, f: TrainFamily) -> Option<&FamilyFit> {
        self.families.iter().find(|x| x.family == f)
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

/// Decide what `hw` can train. `ram_gb`: total system memory (Apple unified memory), if known.
pub fn capability(hw: &HardwareInfo, ram_gb: Option<f64>) -> Capability {
    let mut reasons = Vec::new();
    let accel = if hw.is_apple_silicon() {
        let name = hw
            .gpus_of(GpuVendor::Apple)
            .next()
            .map_or_else(|| "Apple GPU".to_string(), |g| g.name.clone());
        let memory_mb = ram_gb.map_or(0, |g| estimate::to_u64(g * 1024.0 * MPS_MEMORY_SHARE));
        Some(Accelerator::Mps { name, memory_mb })
    } else {
        pick_cuda(hw, &mut reasons)
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
    let families = TrainFamily::ALL
        .into_iter()
        .map(|f| family_fit(f, accel.as_ref()))
        .collect();
    Capability {
        available: accel.is_some(),
        accelerator: accel,
        reasons,
        families,
    }
}

/// The NVIDIA GPU to train on: the qualifying one with the most VRAM. Every GPU passed over adds
/// its reason.
fn pick_cuda(hw: &HardwareInfo, reasons: &mut Vec<String>) -> Option<Accelerator> {
    let nvidia: Vec<_> = hw.gpus_of(GpuVendor::Nvidia).collect();
    if nvidia.is_empty() {
        return None;
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
        return None;
    }
    let Some(driver) = hw.cuda_driver else {
        reasons.push(
            "the NVIDIA driver does not report a CUDA version (driver missing or too old); \
             install a current NVIDIA driver (CUDA 12 or newer)"
                .to_string(),
        );
        return None;
    };
    let cu126 = if hw.is_windows() {
        CU126_WINDOWS
    } else {
        CU126_LINUX
    };
    let mut best: Option<Accelerator> = None;
    for g in nvidia {
        let Some(cuda) = g.cuda else {
            reasons.push(format!("{}: not visible to the CUDA driver", g.name));
            continue;
        };
        let cc = cuda.compute_capability;
        if cc < MIN_CC {
            reasons.push(format!(
                "{}: compute capability {cc} is older than {MIN_CC} (Pascal GTX 10xx)",
                g.name
            ));
            continue;
        }
        let lane = if has_kernels(cu126, cc) {
            if (driver.major, driver.minor) < CU126_MIN_DRIVER {
                reasons.push(format!(
                    "{}: the driver supports CUDA {driver}; PyTorch's CUDA 12.6 build needs a \
                     CUDA 12 driver: update the NVIDIA driver",
                    g.name
                ));
                continue;
            }
            TrainLane::Cu126
        } else if has_kernels(CU130, cc) {
            if (driver.major, driver.minor) < CU130_MIN_DRIVER {
                reasons.push(format!(
                    "{}: compute capability {cc} needs PyTorch's CUDA 13.0 build, which needs a \
                     CUDA 13 driver (this one supports {driver}): update the NVIDIA driver",
                    g.name
                ));
                continue;
            }
            TrainLane::Cu130
        } else {
            reasons.push(format!(
                "{}: the pinned PyTorch 2.14 builds have no kernels for compute capability {cc}",
                g.name
            ));
            continue;
        };
        let cand = Accelerator::Cuda {
            ordinal: cuda.ordinal,
            name: g.name.clone(),
            compute_capability: cc,
            vram_mb: g.vram_mb,
            lane,
        };
        if best
            .as_ref()
            .is_none_or(|b| cand.memory_mb() > b.memory_mb())
        {
            best = Some(cand);
        }
    }
    best
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
