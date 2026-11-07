//! Live GPU memory per adapter: free and total now, never cached past [`MAX_AGE`].
//!
//! - **NVIDIA** (Windows, Linux): NVML (`nvml.dll` / `libnvidia-ml.so.1`, loaded at run time like
//!   the CUDA driver in [`super::detect`]), found by PCI bus id. NVML creates no CUDA context, which
//!   would itself take memory on the card. Free is the device's, whoever holds the rest.
//! - **Windows, any vendor** (and NVIDIA without NVML): DXGI `IDXGIAdapter3::QueryVideoMemoryInfo`,
//!   found by LUID: the budget the OS gives this process minus what it uses. An integrated GPU
//!   (Intel UHD 630) has no memory of its own: its reading is the shared-memory budget, marked
//!   [`MemoryKind::Shared`], never presented as VRAM.
//! - **Linux AMD**: amdgpu's `mem_info_vram_total` / `mem_info_vram_used` in sysfs.
//! - **macOS** (unified memory), Intel on Linux, anything else: no reading. Unknown is None, never
//!   a guess, so nothing warns on it.
//!
//! Beside the readings: which loaded models hold memory on which adapter ([`hold`], kept by the
//! ONNX Runtime backend while its sessions live), the model load line ([`LoadProbe`]), the warning
//! a GPU job gets when the card is short of free memory ([`job_warning`], [`check_cuda_job`]), and
//! the per-session arena cap of ONNX Runtime's CUDA provider (`ort_gpu_mem_limit_mb`).
//!
//! Every decision and parser is a pure function; the `unsafe` FFI is in the `nvml` and `dxgi`
//! modules only.

use super::detect::{self, AdapterIds, GpuAdapter, GpuVendor, HardwareInfo, PciAddress};
use super::spec::{Device, Runtime, Target};
use serde::Serialize;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A reading older than this is taken again ([`readings`]).
pub const MAX_AGE: Duration = Duration::from_secs(2);

const MIB: u64 = 1024 * 1024;

/// What the adapter's figure measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    /// Memory on the card.
    Dedicated,
    /// System memory the OS lets the (integrated) GPU use: the DXGI shared budget.
    Shared,
}

/// Where a reading came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Nvml,
    Dxgi,
    Sysfs,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Nvml => "nvml",
            Source::Dxgi => "dxgi",
            Source::Sysfs => "sysfs",
        }
    }
}

/// The memory of one adapter now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VramReading {
    /// [`GpuAdapter::index`].
    pub index: u32,
    pub name: String,
    pub vendor: GpuVendor,
    pub kind: MemoryKind,
    pub total_mb: u64,
    pub free_mb: u64,
    pub source: Source,
}

impl VramReading {
    pub fn used_mb(&self) -> u64 {
        self.total_mb.saturating_sub(self.free_mb)
    }

    /// `1.9 GB free of 8.0 GB`, or `3.1 GB of the 7.9 GB shared-memory budget free`.
    pub fn text(&self) -> String {
        match self.kind {
            MemoryKind::Dedicated => {
                format!("{} free of {}", gb(self.free_mb), gb(self.total_mb))
            }
            MemoryKind::Shared => format!(
                "{} of the {} shared-memory budget free",
                gb(self.free_mb),
                gb(self.total_mb)
            ),
        }
    }
}

/// `1.9 GB` (one decimal, 1024-based like the cards are sold).
pub fn gb(mb: u64) -> String {
    format!("{:.1} GB", mb as f64 / 1024.0)
}

/// Free and total in MiB from NVML's bytes (`nvmlMemory_t`). Pure.
pub fn from_nvml(total: u64, free: u64) -> (u64, u64) {
    (total / MIB, free.min(total) / MIB)
}

/// One DXGI memory segment group (`DXGI_QUERY_VIDEO_MEMORY_INFO`), in bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Segment {
    pub budget: u64,
    pub usage: u64,
}

/// What DXGI tells about one adapter, in bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DxgiMemory {
    /// `DXGI_ADAPTER_DESC1::DedicatedVideoMemory`.
    pub dedicated: u64,
    pub local: Segment,
    pub non_local: Segment,
}

/// Kind, total and free in MiB of a DXGI reading. A discrete adapter: its dedicated memory, free =
/// the local budget (capped at the card) minus this process's usage. An integrated adapter has no
/// memory of its own (DXGI puts the shared pool in the local group on UMA adapters, in the
/// non-local one elsewhere): the larger budget is the total. None when DXGI reports no budget.
/// Pure.
pub fn from_dxgi(m: &DxgiMemory, integrated: bool) -> Option<(MemoryKind, u64, u64)> {
    if integrated {
        let s = if m.non_local.budget > m.local.budget {
            m.non_local
        } else {
            m.local
        };
        if s.budget == 0 {
            return None;
        }
        return Some((
            MemoryKind::Shared,
            s.budget / MIB,
            s.budget.saturating_sub(s.usage) / MIB,
        ));
    }
    let total = if m.dedicated > 0 {
        m.dedicated
    } else {
        m.local.budget
    };
    if total == 0 || m.local.budget == 0 {
        return None;
    }
    let free = m.local.budget.min(total).saturating_sub(m.local.usage);
    Some((MemoryKind::Dedicated, total / MIB, free / MIB))
}

/// Total and free in MiB from amdgpu's `mem_info_vram_total` / `mem_info_vram_used` (bytes, one
/// number per file). None when either does not parse or the total is 0. Pure.
pub fn from_sysfs(total: &str, used: &str) -> Option<(u64, u64)> {
    let total: u64 = total.trim().parse().ok()?;
    let used: u64 = used.trim().parse().ok()?;
    (total > 0).then(|| (total / MIB, total.saturating_sub(used) / MIB))
}

/// The bus id NVML's `nvmlDeviceGetHandleByPciBusId_v2` takes (`domain:bus:device.function`, hex).
pub fn nvml_bus_id(p: PciAddress) -> String {
    format!(
        "{:08x}:{:02x}:{:02x}.{:x}",
        p.domain, p.bus, p.device, p.function
    )
}

/// The sources to try for `g` on `os`, in order. Pure.
pub fn sources_for(os: &str, g: &GpuAdapter) -> Vec<Source> {
    let mut v = Vec::new();
    if g.vendor == GpuVendor::Nvidia && matches!(os, "windows" | "linux") {
        v.push(Source::Nvml);
    }
    if os == "windows" {
        v.push(Source::Dxgi);
    }
    if os == "linux" && g.vendor == GpuVendor::Amd {
        v.push(Source::Sysfs);
    }
    v
}

/// The adapter (its [`GpuAdapter::index`]) a device spec runs on, when it is a GPU: CUDA and
/// TensorRT by CUDA ordinal, DirectML by DXGI index, OpenVINO `GPU.N` as OpenVINO numbers Intel
/// GPUs (the integrated one first, then the discrete ones in platform order). None for the CPU,
/// the NPU, CoreML (unified memory) and an index no adapter has. Pure.
pub fn adapter_for(hw: &HardwareInfo, device: &Device) -> Option<u32> {
    let n = device.index.unwrap_or(0);
    match (device.runtime, device.target) {
        (Runtime::Ort, Target::Cuda | Target::TensorRt) => adapter_for_cuda(hw, n),
        (Runtime::Ort, Target::DirectMl) => hw.gpus.iter().find(|g| g.index == n).map(|g| g.index),
        (Runtime::OpenVino, Target::Gpu) => {
            let mut intel: Vec<&GpuAdapter> = hw.gpus_of(GpuVendor::Intel).collect();
            intel.sort_by_key(|g| g.discrete);
            intel.get(usize::try_from(n).ok()?).map(|g| g.index)
        }
        _ => None,
    }
}

/// The adapter of CUDA device `ordinal` (`cuda:N` of a job, `ort:cuda:N`). Pure.
pub fn adapter_for_cuda(hw: &HardwareInfo, ordinal: u32) -> Option<u32> {
    hw.gpus
        .iter()
        .find(|g| g.cuda.is_some_and(|c| c.ordinal == ordinal))
        .map(|g| g.index)
}

// ---- readings ----------------------------------------------------------------------------------

/// Every adapter with a reading, at most [`MAX_AGE`] old.
pub fn readings() -> Vec<VramReading> {
    static CACHE: Mutex<Option<(Instant, Vec<VramReading>)>> = Mutex::new(None);
    let mut c = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((at, v)) = c.as_ref()
        && at.elapsed() < MAX_AGE
    {
        return v.clone();
    }
    let v = read_all();
    *c = Some((Instant::now(), v.clone()));
    v
}

/// The reading of adapter `index` (at most [`MAX_AGE`] old).
pub fn reading(index: u32) -> Option<VramReading> {
    readings().into_iter().find(|r| r.index == index)
}

/// The reading of adapter `index` taken now (the load line's before and after).
pub fn reading_now(index: u32) -> Option<VramReading> {
    let hw = detect::hardware();
    let ids = detect::adapter_ids();
    let at = hw.gpus.iter().position(|g| g.index == index)?;
    let g = hw.gpus.get(at)?;
    read_one(hw, g, ids.get(at).copied().unwrap_or_default())
}

fn read_all() -> Vec<VramReading> {
    let hw = detect::hardware();
    let ids = detect::adapter_ids();
    hw.gpus
        .iter()
        .enumerate()
        .filter_map(|(i, g)| read_one(hw, g, ids.get(i).copied().unwrap_or_default()))
        .collect()
}

fn read_one(hw: &HardwareInfo, g: &GpuAdapter, ids: AdapterIds) -> Option<VramReading> {
    let nvidia = hw.gpus_of(GpuVendor::Nvidia).count();
    for source in sources_for(&hw.os, g) {
        let got = match source {
            Source::Nvml => {
                platform::nvml(ids.pci, nvidia == 1).map(|(t, f)| (MemoryKind::Dedicated, t, f))
            }
            Source::Dxgi => ids.luid.and_then(|l| platform::dxgi(l, !g.discrete)),
            Source::Sysfs => platform::sysfs(g.index).map(|(t, f)| (MemoryKind::Dedicated, t, f)),
        };
        if let Some((kind, total_mb, free_mb)) = got {
            return Some(VramReading {
                index: g.index,
                name: g.name.clone(),
                vendor: g.vendor,
                kind,
                total_mb,
                free_mb,
                source,
            });
        }
    }
    None
}

#[cfg(any(windows, target_os = "linux"))]
mod platform {
    use super::*;

    /// NVML's total and free (MiB) of the GPU at `pci`, or of the only GPU when `only_one` and the
    /// address is unknown.
    pub fn nvml(pci: Option<PciAddress>, only_one: bool) -> Option<(u64, u64)> {
        let (total, free) = super::nvml::memory(pci, only_one)?;
        Some(from_nvml(total, free))
    }

    #[cfg(windows)]
    pub fn dxgi(luid: u64, integrated: bool) -> Option<(MemoryKind, u64, u64)> {
        match super::dxgi::query(luid) {
            Ok(m) => from_dxgi(&m, integrated),
            Err(e) => {
                tracing::debug!("DXGI memory query of adapter {luid:#x} failed: {e}");
                None
            }
        }
    }

    #[cfg(not(windows))]
    pub fn dxgi(_luid: u64, _integrated: bool) -> Option<(MemoryKind, u64, u64)> {
        None
    }

    #[cfg(target_os = "linux")]
    pub fn sysfs(card: u32) -> Option<(u64, u64)> {
        sysfs_in(std::path::Path::new("/sys/class/drm"), card)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn sysfs(_card: u32) -> Option<(u64, u64)> {
        None
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod platform {
    use super::*;

    pub fn nvml(_pci: Option<PciAddress>, _only_one: bool) -> Option<(u64, u64)> {
        None
    }

    pub fn dxgi(_luid: u64, _integrated: bool) -> Option<(MemoryKind, u64, u64)> {
        None
    }

    pub fn sysfs(_card: u32) -> Option<(u64, u64)> {
        None
    }
}

/// [`from_sysfs`] of DRM card `card` under `root` (`/sys/class/drm`).
#[cfg(any(target_os = "linux", test))]
pub(crate) fn sysfs_in(root: &std::path::Path, card: u32) -> Option<(u64, u64)> {
    let dev = root.join(format!("card{card}")).join("device");
    let read = |f: &str| std::fs::read_to_string(dev.join(f)).ok();
    from_sysfs(&read("mem_info_vram_total")?, &read("mem_info_vram_used")?)
}

/// NVML (`nvml.dll` / `libnvidia-ml.so.1`): loaded and initialized once, never unloaded.
#[cfg(any(windows, target_os = "linux"))]
mod nvml {
    use super::*;
    use std::ffi::{CString, c_char, c_int, c_uint, c_void};
    use std::sync::OnceLock;

    type NvmlReturn = c_int;
    type NvmlDevice = *mut c_void;
    const NVML_SUCCESS: NvmlReturn = 0;

    /// `nvmlMemory_t` (v1): bytes.
    #[repr(C)]
    #[derive(Default)]
    struct NvmlMemory {
        total: u64,
        free: u64,
        #[allow(dead_code, reason = "part of the C layout NVML writes")]
        used: u64,
    }

    #[cfg(windows)]
    const LIBRARY: &str = "nvml.dll";
    #[cfg(not(windows))]
    const LIBRARY: &str = "libnvidia-ml.so.1";

    type Init = unsafe extern "C" fn() -> NvmlReturn;
    type GetCount = unsafe extern "C" fn(*mut c_uint) -> NvmlReturn;
    type ByBusId = unsafe extern "C" fn(*const c_char, *mut NvmlDevice) -> NvmlReturn;
    type ByIndex = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn;
    type MemoryInfo = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> NvmlReturn;

    struct Api {
        count: GetCount,
        by_bus: ByBusId,
        by_index: ByIndex,
        memory: MemoryInfo,
    }

    fn api() -> Option<&'static Api> {
        static API: OnceLock<Option<Api>> = OnceLock::new();
        API.get_or_init(|| match load() {
            Ok(a) => Some(a),
            Err(e) => {
                tracing::debug!("NVML unavailable, no live NVIDIA memory reading from it: {e}");
                None
            }
        })
        .as_ref()
    }

    fn load() -> Result<Api, String> {
        // SAFETY: loads the NVIDIA management library the driver installs (System32 / the
        // driver's ld.so.conf dir, never the working directory); its initializers have no
        // preconditions.
        let lib = unsafe { crate::backend::libs::open_system(LIBRARY) }
            .map_err(|e| format!("{LIBRARY}: {e}"))?;
        // Never unloaded: the function pointers below live for the process.
        let lib: &'static libloading::Library = Box::leak(Box::new(lib));
        let sym = |name: &[u8]| -> Result<*const (), String> {
            // SAFETY: `lib` stays loaded for the whole process (leaked above); the symbol is
            // read as a plain code pointer and cast to its nvml.h signature below.
            unsafe { crate::backend::libs::symbol::<*const ()>(lib, name) }
                .map(|s| *s)
                .map_err(|e| format!("{LIBRARY}: {e}"))
        };
        // SAFETY: each pointer comes from the symbol of that name and is transmuted to its
        // nvml.h signature.
        let (init, count, by_bus, by_index, memory) = unsafe {
            (
                std::mem::transmute::<*const (), Init>(sym(b"nvmlInit_v2\0")?),
                std::mem::transmute::<*const (), GetCount>(sym(b"nvmlDeviceGetCount_v2\0")?),
                std::mem::transmute::<*const (), ByBusId>(sym(
                    b"nvmlDeviceGetHandleByPciBusId_v2\0",
                )?),
                std::mem::transmute::<*const (), ByIndex>(sym(b"nvmlDeviceGetHandleByIndex_v2\0")?),
                std::mem::transmute::<*const (), MemoryInfo>(sym(b"nvmlDeviceGetMemoryInfo\0")?),
            )
        };
        // SAFETY: plain initialization call, no arguments.
        let r = unsafe { init() };
        if r != NVML_SUCCESS {
            return Err(format!("nvmlInit_v2 returned {r}"));
        }
        Ok(Api {
            count,
            by_bus,
            by_index,
            memory,
        })
    }

    /// Total and free bytes of the GPU at `pci` (or the only one).
    pub fn memory(pci: Option<PciAddress>, only_one: bool) -> Option<(u64, u64)> {
        let api = api()?;
        let mut dev: NvmlDevice = std::ptr::null_mut();
        let found = match pci {
            Some(p) => {
                let id = CString::new(nvml_bus_id(p)).ok()?;
                // SAFETY: a NUL-terminated string and a local out-parameter.
                unsafe { (api.by_bus)(id.as_ptr(), &mut dev) }
            }
            None if only_one => {
                let mut n: c_uint = 0;
                // SAFETY: a local out-parameter.
                let r = unsafe { (api.count)(&mut n) };
                if r != NVML_SUCCESS || n != 1 {
                    return None;
                }
                // SAFETY: index 0 of a count of 1, a local out-parameter.
                unsafe { (api.by_index)(0, &mut dev) }
            }
            None => return None,
        };
        if found != NVML_SUCCESS || dev.is_null() {
            return None;
        }
        let mut m = NvmlMemory::default();
        // SAFETY: a handle NVML returned and a local struct of the v1 layout.
        let r = unsafe { (api.memory)(dev, &mut m) };
        (r == NVML_SUCCESS && m.total > 0).then_some((m.total, m.free))
    }
}

/// DXGI 1.4 memory queries (Windows 10+).
#[cfg(windows)]
mod dxgi {
    use super::{DxgiMemory, Segment};
    use windows::Win32::Foundation::LUID;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
        IDXGIFactory4,
    };

    pub fn query(luid: u64) -> windows::core::Result<DxgiMemory> {
        // SAFETY: plain COM factory creation; the interface is reference counted.
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }?;
        let luid = LUID {
            LowPart: u32::try_from(luid & 0xFFFF_FFFF).unwrap_or_default(),
            HighPart: u32::try_from(luid >> 32).unwrap_or_default().cast_signed(),
        };
        // SAFETY: a valid factory; the LUID is passed by value.
        let adapter: IDXGIAdapter3 = unsafe { factory.EnumAdapterByLuid(luid) }?;
        // SAFETY: a valid adapter.
        let desc = unsafe { adapter.GetDesc1() }?;
        let segment = |group: DXGI_MEMORY_SEGMENT_GROUP| {
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            // SAFETY: node 0 exists on every adapter; `info` is a local of the documented layout.
            unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }
                .map(|()| Segment {
                    budget: info.Budget,
                    usage: info.CurrentUsage,
                })
                .unwrap_or_default()
        };
        Ok(DxgiMemory {
            dedicated: desc.DedicatedVideoMemory as u64,
            local: segment(DXGI_MEMORY_SEGMENT_GROUP_LOCAL),
            non_local: segment(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL),
        })
    }
}

// ---- who holds what ----------------------------------------------------------------------------

/// A loaded model holding memory on an adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Holder {
    /// The model file's stem (`IPcam-general`).
    pub model: String,
    /// The device spec it runs on (`ort:cuda:0`).
    pub device: String,
    pub adapter: u32,
    /// What its load took ([`LoadProbe`]); None when the card could not be read.
    pub held_mb: Option<u64>,
}

static HOLDERS: Mutex<Vec<(u64, Holder)>> = Mutex::new(Vec::new());
static NEXT_HOLD: AtomicU64 = AtomicU64::new(1);

/// Registration of a [`Holder`], removed when dropped.
#[derive(Debug)]
pub struct Hold(u64);

impl Drop for Hold {
    fn drop(&mut self) {
        let mut h = HOLDERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        h.retain(|(id, _)| *id != self.0);
    }
}

/// Register `h` until the returned [`Hold`] drops.
pub fn hold(h: Holder) -> Hold {
    let id = NEXT_HOLD.fetch_add(1, Ordering::Relaxed);
    HOLDERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push((id, h));
    Hold(id)
}

/// The models holding memory on `adapter` now, in load order.
pub fn holders_on(adapter: u32) -> Vec<Holder> {
    holders()
        .into_iter()
        .filter(|h| h.adapter == adapter)
        .collect()
}

/// Every registered holder, in load order.
pub fn holders() -> Vec<Holder> {
    HOLDERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(_, h)| h.clone())
        .collect()
}

// ---- the load line -----------------------------------------------------------------------------

/// The memory of a model's GPU before it loads, for the load line ([`Self::finish`]).
#[derive(Debug)]
pub struct LoadProbe {
    model: String,
    device: String,
    adapter: Option<u32>,
    before: Option<VramReading>,
}

impl LoadProbe {
    /// Read the GPU `device` runs on (nothing for a CPU, NPU or CoreML device).
    pub fn start(model: &std::path::Path, device: &Device) -> Self {
        let adapter = adapter_for(detect::hardware(), device);
        Self {
            model: model
                .file_stem()
                .map_or_else(String::new, |s| s.to_string_lossy().into_owned()),
            device: device.to_string(),
            adapter,
            before: adapter.and_then(reading_now),
        }
    }

    /// Read the GPU again and log the load line; the adapter and the reading.
    fn measure(&self) -> Option<(u32, Option<VramReading>)> {
        let adapter = self.adapter?;
        let after = reading_now(adapter);
        if let Some(line) = load_line(
            &self.model,
            &self.device,
            self.before.as_ref(),
            after.as_ref(),
        ) {
            tracing::info!(model = %self.model, device = %self.device, "{line}");
        }
        Some((adapter, after))
    }

    /// [`Self::finish`] without the registration (a runtime whose models no GPU job shares a
    /// card with: OpenVINO on Intel GPUs).
    pub fn log(self) {
        let _ = self.measure();
    }

    /// Read the GPU again, log the load line and register the model as a [`Holder`] of what its
    /// load took. None for a device that is no GPU.
    pub fn finish(self) -> Option<Hold> {
        let (adapter, after) = self.measure()?;
        Some(hold(Holder {
            held_mb: taken_mb(self.before.as_ref(), after.as_ref()),
            model: self.model,
            device: self.device,
            adapter,
        }))
    }
}

/// What a load took: free before minus free after (0 when it rose: another program freed
/// memory meanwhile). Pure.
pub fn taken_mb(before: Option<&VramReading>, after: Option<&VramReading>) -> Option<u64> {
    Some(before?.free_mb.saturating_sub(after?.free_mb))
}

/// The load line: `model IPcam-general on ort:cuda:0: GPU memory 7.1 GB free of 8.0 GB before,
/// 6.6 GB after (about 0.5 GB taken; NVIDIA GeForce GTX 1070 Ti, nvml)`. None when neither
/// reading exists. Pure.
pub fn load_line(
    model: &str,
    device: &str,
    before: Option<&VramReading>,
    after: Option<&VramReading>,
) -> Option<String> {
    let any = after.or(before)?;
    let what = match any.kind {
        MemoryKind::Dedicated => "GPU memory",
        MemoryKind::Shared => "GPU shared memory",
    };
    let before_text = before.map_or_else(|| "unknown".to_string(), VramReading::text);
    let after_text = after.map_or_else(|| "unknown".to_string(), |a| gb(a.free_mb));
    let taken = taken_mb(before, after)
        .map(|t| format!("about {} taken; ", gb(t)))
        .unwrap_or_default();
    Some(format!(
        "model {model} on {device}: {what} {before_text} before loading, {after_text} free after \
         ({taken}{}, {})",
        any.name,
        any.source.as_str()
    ))
}

// ---- the job warning ---------------------------------------------------------------------------

/// What a GPU job (training, auto-label, an import's embeddings) expects to need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobNeed {
    /// `this training run`, `this auto-label run`.
    pub what: String,
    pub need_mb: u64,
    /// How the figure was reached: `YOLO26 s, batch 16 at 640 px`.
    pub detail: String,
    /// The smaller choice that would help: `a smaller size or batch`.
    pub smaller: String,
}

/// The warning before a job, or None: the card's free memory covers the need, or it cannot be read
/// (no guess). Names the loaded models on that card and what they took. Never a refusal: the
/// estimate is rough and the job may still fit. Pure.
pub fn job_warning(need: &JobNeed, r: Option<&VramReading>, holders: &[Holder]) -> Option<String> {
    let r = r?;
    if r.free_mb >= need.need_mb {
        return None;
    }
    let models = if holders.is_empty() {
        "No model of this service is loaded on it, so other programs hold the rest: close them \
         for the run"
            .to_string()
    } else {
        let names: Vec<String> = holders
            .iter()
            .map(|h| match h.held_mb {
                Some(mb) if mb > 0 => format!("{} ({}, about {})", h.model, h.device, gb(mb)),
                _ => format!("{} ({})", h.model, h.device),
            })
            .collect();
        format!(
            "Models of this service loaded on it: {}. Disable them or move them to another \
             device for the run (Config page, Models, Device)",
            names.join(", ")
        )
    };
    Some(format!(
        "{} has {} and {} needs about {} ({}): it may run out of GPU memory, or slow down a lot as \
         the driver spills into system memory. {models}, or choose {}.",
        r.name,
        r.text(),
        need.what,
        gb(need.need_mb),
        need.detail,
        need.smaller
    ))
}

/// [`job_warning`] for a job on CUDA device `ordinal`, against the card as it is now.
pub fn check_cuda_job(ordinal: u32, need: &JobNeed) -> Option<String> {
    let adapter = adapter_for_cuda(detect::hardware(), ordinal)?;
    job_warning(need, reading_now(adapter).as_ref(), &holders_on(adapter))
}

// ---- ONNX Runtime's CUDA arena cap -------------------------------------------------------------

static ORT_SESSION_LIMIT_MB: AtomicU64 = AtomicU64::new(0);

/// Set the `gpu_mem_limit` of every later ONNX Runtime CUDA / TensorRT session
/// (config `ort_gpu_mem_limit_mb`; None = no cap).
pub fn set_ort_session_limit_mb(mb: Option<u64>) {
    ORT_SESSION_LIMIT_MB.store(mb.unwrap_or(0), Ordering::Relaxed);
}

/// [`set_ort_session_limit_mb`]'s value.
pub fn ort_session_limit_mb() -> Option<u64> {
    Some(ORT_SESSION_LIMIT_MB.load(Ordering::Relaxed)).filter(|&mb| mb > 0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::backend::detect::{ComputeCapability, CudaInfo, adapter_from_ids};

    fn reading(free_mb: u64, total_mb: u64, kind: MemoryKind) -> VramReading {
        VramReading {
            index: 0,
            name: "NVIDIA GeForce GTX 1070 Ti".into(),
            vendor: GpuVendor::Nvidia,
            kind,
            total_mb,
            free_mb,
            source: Source::Nvml,
        }
    }

    fn nvidia(index: u32, ordinal: u32) -> GpuAdapter {
        let mut g = adapter_from_ids(index, detect::VENDOR_NVIDIA, 0x1B82, Some("GTX"), 8192);
        g.cuda = Some(CudaInfo {
            ordinal,
            compute_capability: ComputeCapability::new(6, 1),
        });
        g
    }

    #[test]
    fn nvml_bytes_and_bus_ids() {
        assert_eq!(from_nvml(8 * 1024 * MIB, 3 * 1024 * MIB + 5), (8192, 3072));
        // A free figure above the total (never seen) is capped.
        assert_eq!(from_nvml(MIB * 10, MIB * 20), (10, 10));
        let p = PciAddress::parse("0000:01:00.0").unwrap();
        assert_eq!(nvml_bus_id(p), "00000000:01:00.0");
        assert_eq!(
            PciAddress::parse(&nvml_bus_id(p)),
            Some(p),
            "NVML's own format parses back"
        );
    }

    #[test]
    fn dxgi_dedicated_and_shared() {
        let gib = 1024 * MIB;
        // A discrete card: the dedicated memory is the total, free = budget - usage.
        let m = DxgiMemory {
            dedicated: 8 * gib,
            local: Segment {
                budget: 7 * gib,
                usage: 2 * gib,
            },
            non_local: Segment {
                budget: 16 * gib,
                usage: 0,
            },
        };
        assert_eq!(
            from_dxgi(&m, false),
            Some((MemoryKind::Dedicated, 8192, 5120))
        );
        // A budget above the card (never trusted) is capped at the card.
        let big = DxgiMemory {
            local: Segment {
                budget: 9 * gib,
                usage: 0,
            },
            ..m
        };
        assert_eq!(from_dxgi(&big, false).unwrap().2, 8192);
        // UHD 630: 128 MB carve-out, the shared pool in the local group (UMA).
        let igpu = DxgiMemory {
            dedicated: 128 * MIB,
            local: Segment {
                budget: 8 * gib,
                usage: gib,
            },
            non_local: Segment::default(),
        };
        assert_eq!(
            from_dxgi(&igpu, true),
            Some((MemoryKind::Shared, 8192, 7168))
        );
        // ... or in the non-local group: the larger budget wins.
        let other = DxgiMemory {
            local: Segment {
                budget: 128 * MIB,
                usage: 0,
            },
            non_local: Segment {
                budget: 6 * gib,
                usage: 3 * gib,
            },
            ..igpu
        };
        assert_eq!(
            from_dxgi(&other, true),
            Some((MemoryKind::Shared, 6144, 3072))
        );
        // No budget at all: no reading.
        assert_eq!(from_dxgi(&DxgiMemory::default(), true), None);
        assert_eq!(
            from_dxgi(
                &DxgiMemory {
                    dedicated: gib,
                    ..DxgiMemory::default()
                },
                false
            ),
            None
        );
    }

    #[test]
    fn sysfs_files() {
        assert_eq!(
            from_sysfs("17163091968\n", "1073741824\n"),
            Some((16368, 15344))
        );
        assert_eq!(from_sysfs("0", "0"), None);
        assert_eq!(from_sysfs("x", "1"), None);
        // Used above total (racing reads) is 0 free, not a wrap.
        assert_eq!(from_sysfs("1048576", "2097152"), Some((1, 0)));
        struct Tmp(std::path::PathBuf);
        impl Drop for Tmp {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let dir = Tmp(std::env::temp_dir().join(format!("bop-vram-{}", uuid::Uuid::new_v4())));
        let dev = dir.0.join("card1").join("device");
        std::fs::create_dir_all(&dev).unwrap();
        std::fs::write(dev.join("mem_info_vram_total"), "8589934592\n").unwrap();
        std::fs::write(dev.join("mem_info_vram_used"), "536870912\n").unwrap();
        assert_eq!(sysfs_in(&dir.0, 1), Some((8192, 7680)));
        assert_eq!(sysfs_in(&dir.0, 0), None);
    }

    #[test]
    fn sources_per_platform_and_vendor() {
        let nv = nvidia(0, 0);
        let intel = adapter_from_ids(1, detect::VENDOR_INTEL, 0x3E92, Some("UHD 630"), 128);
        let amd = adapter_from_ids(2, detect::VENDOR_AMD, 0x73BF, Some("RX 6800"), 16368);
        assert_eq!(
            sources_for("windows", &nv),
            [Source::Nvml, Source::Dxgi],
            "NVML first, DXGI when it cannot be loaded"
        );
        assert_eq!(sources_for("linux", &nv), [Source::Nvml]);
        assert_eq!(sources_for("windows", &intel), [Source::Dxgi]);
        assert!(sources_for("linux", &intel).is_empty(), "no guess");
        assert_eq!(sources_for("linux", &amd), [Source::Sysfs]);
        assert_eq!(sources_for("windows", &amd), [Source::Dxgi]);
        assert!(sources_for("macos", &detect::apple_gpu("Apple M1")).is_empty());
    }

    #[test]
    fn devices_map_to_adapters() {
        let uhd = adapter_from_ids(0, detect::VENDOR_INTEL, 0x3E92, Some("UHD 630"), 128);
        let arc = adapter_from_ids(2, detect::VENDOR_INTEL, 0x56A0, Some("Arc A770"), 16000);
        let hw = HardwareInfo::new(
            "windows",
            "x86_64",
            vec![uhd, nvidia(1, 0), arc, nvidia(3, 1)],
        );
        let dev = |s: &str| match crate::backend::spec::parse(s).unwrap() {
            crate::backend::spec::DeviceSpec::Device(d) => d,
            crate::backend::spec::DeviceSpec::Auto => unreachable!(),
        };
        assert_eq!(adapter_for(&hw, &dev("ort:cuda")), Some(1));
        assert_eq!(adapter_for(&hw, &dev("ort:cuda:1")), Some(3));
        assert_eq!(adapter_for(&hw, &dev("ort:tensorrt:1")), Some(3));
        assert_eq!(adapter_for(&hw, &dev("ort:cuda:2")), None);
        assert_eq!(adapter_for(&hw, &dev("ort:directml:2")), Some(2));
        assert_eq!(adapter_for(&hw, &dev("ort:directml:7")), None);
        // OpenVINO: GPU.0 is the integrated GPU, GPU.1 the Arc.
        assert_eq!(adapter_for(&hw, &dev("openvino:gpu")), Some(0));
        assert_eq!(adapter_for(&hw, &dev("openvino:gpu.1")), Some(2));
        assert_eq!(adapter_for(&hw, &dev("openvino:gpu.2")), None);
        for none in ["openvino:cpu", "ort:cpu", "ort:coreml", "openvino:npu"] {
            assert_eq!(adapter_for(&hw, &dev(none)), None, "{none}");
        }
        assert_eq!(adapter_for_cuda(&hw, 0), Some(1));
    }

    #[test]
    fn texts_and_the_load_line() {
        let before = reading(7270, 8192, MemoryKind::Dedicated);
        let after = reading(6758, 8192, MemoryKind::Dedicated);
        assert_eq!(before.text(), "7.1 GB free of 8.0 GB");
        assert_eq!(before.used_mb(), 922);
        assert_eq!(
            load_line("IPcam-general", "ort:cuda:0", Some(&before), Some(&after)).unwrap(),
            "model IPcam-general on ort:cuda:0: GPU memory 7.1 GB free of 8.0 GB before loading, \
             6.6 GB free after (about 0.5 GB taken; NVIDIA GeForce GTX 1070 Ti, nvml)"
        );
        // Free rose during the load: 0 taken, not a wrap.
        assert_eq!(taken_mb(Some(&after), Some(&before)), Some(0));
        let line = load_line(
            "m",
            "openvino:gpu",
            None,
            Some(&reading(3000, 8000, MemoryKind::Shared)),
        )
        .unwrap();
        assert!(
            line.contains("GPU shared memory unknown before loading, 2.9 GB free after ("),
            "{line}"
        );
        assert_eq!(load_line("m", "ort:cuda", None, None), None);
        assert_eq!(
            reading(3000, 8000, MemoryKind::Shared).text(),
            "2.9 GB of the 7.8 GB shared-memory budget free"
        );
        let json = serde_json::to_value(&before).unwrap();
        assert_eq!(json["kind"], "dedicated");
        assert_eq!(json["source"], "nvml");
        assert_eq!(json["freeMb"], 7270);
        assert_eq!(json["vendor"], "NVIDIA");
    }

    #[test]
    fn job_warning_only_when_the_card_is_short() {
        let need = JobNeed {
            what: "this training run".into(),
            need_mb: 5000,
            detail: "YOLO26 s, batch 16 at 640 px".into(),
            smaller: "a smaller size or batch".into(),
        };
        let hold = |model: &str, mb| Holder {
            model: model.into(),
            device: "ort:cuda:0".into(),
            adapter: 0,
            held_mb: mb,
        };
        // Unknown free memory: no warning, not a guess.
        assert_eq!(job_warning(&need, None, &[]), None);
        // Enough free.
        let fine = reading(6000, 8192, MemoryKind::Dedicated);
        assert_eq!(
            job_warning(&need, Some(&fine), &[hold("a", Some(500))]),
            None
        );
        let short = reading(2048, 8192, MemoryKind::Dedicated);
        let w = job_warning(
            &need,
            Some(&short),
            &[hold("IPcam-general", Some(600)), hold("yolo26s", None)],
        )
        .unwrap();
        assert!(
            w.starts_with(
                "NVIDIA GeForce GTX 1070 Ti has 2.0 GB free of 8.0 GB and this training run needs \
                 about 4.9 GB (YOLO26 s, batch 16 at 640 px)"
            ),
            "{w}"
        );
        assert!(
            w.contains(
                "loaded on it: IPcam-general (ort:cuda:0, about 0.6 GB), yolo26s (ort:cuda:0)."
            ),
            "{w}"
        );
        assert!(w.ends_with("or choose a smaller size or batch."), "{w}");
        let alone = job_warning(&need, Some(&short), &[]).unwrap();
        assert!(alone.contains("other programs hold the rest"), "{alone}");
    }

    #[test]
    fn holds_register_until_dropped() {
        let h = hold(Holder {
            model: "vram-test-model".into(),
            device: "ort:cuda:5".into(),
            adapter: 905,
            held_mb: Some(10),
        });
        assert_eq!(holders_on(905).len(), 1);
        assert_eq!(holders_on(905)[0].model, "vram-test-model");
        drop(h);
        assert!(holders_on(905).is_empty());
    }

    #[test]
    fn session_limit_round_trips() {
        set_ort_session_limit_mb(Some(1536));
        assert_eq!(ort_session_limit_mb(), Some(1536));
        set_ort_session_limit_mb(None);
        assert_eq!(ort_session_limit_mb(), None);
        set_ort_session_limit_mb(Some(0));
        assert_eq!(ort_session_limit_mb(), None);
    }

    #[test]
    fn readings_on_this_machine_never_panic_and_are_consistent() {
        let all = readings();
        for r in &all {
            assert!(r.free_mb <= r.total_mb, "{r:?}");
        }
        // The cache answers again within MAX_AGE.
        assert_eq!(readings().len(), all.len());
        #[cfg(target_os = "macos")]
        assert!(all.is_empty(), "unified memory: no reading on macOS");
    }
}
